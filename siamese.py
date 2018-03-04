import tensorflow as tf
import numpy as np

class ShapeError(Exception):
	pass


# Network that takes some image with (shape) and tells whether the two inputs are of the same class
class SiameseNetwork(object):
	def __init__(self, input_shape, margin=5.0, learning_rate=1e-3):
		self.input_shape = input_shape
		self.margin = margin

		self.inputs_left, self.inputs_right, self.embedding_left, self.embedding_right = self.create_network()


		self.labels, self.distance, self.loss = self.create_loss()


		self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


	def create_siamese_part(self, inputs, reuse):
		with tf.variable_scope('siamese', reuse=reuse):

			if len(self.input_shape) == 1:
				pass

				net = tf.layers.dense(inputs, 1024, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu)
				net = tf.layers.dense(net, 1024, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu)
				net = tf.layers.dense(net, 2)

			else:

				if len(self.input_shape) == 2:
					# add channel dimension of 1
					inputs = tf.expand_dims(inputs, axis=3)

				elif len(self.input_shape) == 3:
					# no change necessary
					pass
				else:
					# Undefined
					raise ShapeError('Cannot init network with shape ' + self.input_shape)

				# pass through conv2d layers

				# (?, 105, 105, 1/3)

				num_filters = 15

				net = tf.layers.conv2d(inputs, num_filters, 6, strides=1, padding='valid', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
				# (?, 100, 100, 15)

				net = tf.layers.max_pooling2d(net, 3, strides=3)
				net = tf.nn.relu(net)
				# (?, 33, 33, 15)
				print('A', net)

				# second conv layer
				net = tf.layers.conv2d(net, num_filters * 2, 6, strides=1, padding='valid', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
				# (?, 25, 25, 30)

				net = tf.layers.max_pooling2d(net, 3, strides=3)
				net = tf.nn.relu(net)
				# (?, 8, 8, 30)

				print('B', net)


				# third conv layer
				net = tf.layers.conv2d(net, num_filters * 4, 3, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
				# (?, 9, 9, 60)

				print('C', net)

				net = tf.layers.max_pooling2d(net, 9, strides=9)
				net = tf.nn.relu(net)
				# (?, 1, 1, 60)
				print('D', net)


				net = tf.reshape(net, [-1, np.prod(net.shape[1:])])

				net = tf.layers.dense(net, 2)

			return net

	def create_network(self, reuse=False):
		with tf.variable_scope('siamese_network', reuse=reuse):
			inputs_left = tf.placeholder(tf.float32, [None] + list(self.input_shape))
			inputs_right = tf.placeholder(tf.float32, [None] + list(self.input_shape))

			network_left = self.create_siamese_part(inputs_left, reuse=reuse)
			network_right = self.create_siamese_part(inputs_right, reuse=True)

			print(network_left)

			
			return inputs_left, inputs_right, network_left, network_right


	def create_loss(self):

		labels = tf.placeholder(tf.float32, [None])

		# calculate euclidian distance
		# add a small amount to avoid nans
		distance = tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(self.embedding_left - self.embedding_right, 2), axis=1))

		print(self.embedding_left, distance)


		# In the original formula: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
		# Y = 0 Corresponds to "Same", Y = 1 Corresponds to "different"
		# (1 - Y) * 1/2 * distance^2 + (Y) * 1/2 * (max(0, m - distance))^2
		# In this implementation, y = 1 means "same", and y = 0 means "different"
		# So we flip (1 - Y) and Y
		m = tf.constant(float(self.margin), tf.float32)
		#losses = (1 - labels) * 0.5 * tf.pow(distance, 2) + labels * 0.5 * tf.pow(tf.maximum(0.0, m - distance), 2)
		losses = labels * 0.5 * tf.pow(distance, 2) + (1 - labels) * 0.5 * tf.pow(tf.maximum(0.0, m - distance), 2)

		loss = tf.reduce_mean(losses)

		return labels, distance, loss
		
	def train(self, inputs_left, inputs_right, labels):
		sess = tf.get_default_session()

		loss, _, distance = sess.run([self.loss, self.optimize, self.distance], feed_dict={
			self.inputs_left: inputs_left,
			self.inputs_right: inputs_right,
			self.labels: labels	
		})

		return loss, distance


	def get_embedding(self, inputs):
		sess = tf.get_default_session()

		embedding = sess.run(self.embedding_left, feed_dict={
			self.inputs_left: inputs
		})

		return embedding

	def get_distance(self, inputs_left, inputs_right):
		sess = tf.get_default_session()

		distance = sess.run(self.distance, feed_dict={
			self.inputs_left: inputs_left,
			self.inputs_right: inputs_right
		})

		return distance


if __name__ == '__main__':

	import argparse
	import time
	from omniglot import Omniglot
	import logging
	import os

	from visualize import visualize

	logging.basicConfig(level=logging.INFO)

	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG) # set to INFO if you want fewer messages

	parser = argparse.ArgumentParser()
	parser.add_argument('--logdir', type=str, default='events/%d' % int(time.time() * 1000), help='Directory where checkpoint and summary is stored')
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--test', action='store_true')
	parser.add_argument('--train-steps', default=100000, help='Number of training steps')
	parser.add_argument('--batch-size', default=64, help='Size of the minibatch')
	parser.add_argument('--learning-rate', default=1e-3, help='Default learning rate')

	args = parser.parse_args()

	omniglot_shape = (105, 105)

	batch_size = args.batch_size
	batch_size_2 = int(batch_size / 2)

	num_classes = 5
	num_samples = 20


	train_steps = int(args.train_steps)
	print_every_n = 1000


	og = Omniglot(omniglot_shape)

	network = SiameseNetwork(omniglot_shape, learning_rate=args.learning_rate)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		if args.logdir is not None and os.path.exists(args.logdir):
			checkpoint_state = tf.train.get_checkpoint_state(args.logdir)
			if checkpoint_state is not None:
				try:
					saver.restore(sess, checkpoint_state.model_checkpoint_path)
					logger.info('Restoring previous session')
				except (tf.errors.NotFoundError):
					logger.info('Could not find checkpoint at %s', checkpoint_state.model_checkpoint_path)


		if args.train:

			for i in range(train_steps):

				x, y = og.TrainBatch(batch_size, classes=num_classes, samples=num_samples, flatten=False)

				x_left = x[:batch_size_2]
				x_right = x[batch_size_2:]

				y_left = np.array(y[:batch_size_2])
				y_right = np.array(y[batch_size_2:])

				labels = np.all(y_left == y_right, axis=1).astype(np.float32)


				loss, distance = network.train(x_left, x_right, labels)

				if i % print_every_n == 0:
					print(i, loss)

					if not os.path.isdir(args.logdir):
						os.makedirs(args.logdir)

					saver.save(sess, os.path.join(args.logdir, 'model.ckpt'), i)

			saver.save(sess, os.path.join(args.logdir, 'model.ckpt'), train_steps)

		if args.test:

			x, y = og.TestBatch(200, classes=10, samples=20, one_hot=False, flatten=False)

			embeds = network.get_embedding(x)

			# Invert colors and set all non-black pixels to pure white to display nicely in graph
			x = ((255 - np.array(x)) > 0).astype(np.float32)

			visualize(embeds, x, y)