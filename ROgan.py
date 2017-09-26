import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import shutil
import argparse
import time
import cPickle as pickle
import sys
import math

import logging
#import cProfile as profile

from tensorflow.examples.tutorials.mnist import input_data

from scipy.misc import imresize, imsave, imread
from skimage.color import rgb2gray

from ops import *
#from dcgan_ops import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_loader import Loader

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # set to INFO if you want fewer messages


def plot(samples, h, w, c):
	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(8, 8)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		if c == 1:
			plt.imshow(sample.reshape(h, w), cmap='gray')
		else:
			plt.imshow(sample.reshape(h, w, c))
			#plt.imshow(sample)

	return fig


class DCGAN(object):
	def __init__(self, observation_space_d, observation_space_g, output_space, d_layers=4, g_layers=4, learning_rate=0.0002, momentum=0.5):
		self.observation_space_d = observation_space_d
		self.observation_space_g = observation_space_g
		self.output_space = output_space
		self.d_layers = d_layers
		self.g_layers = g_layers

		self.learning_rate = learning_rate
		self.momentum = momentum

		self.inputs = tf.placeholder(tf.float32, [None, np.sum(self.observation_space_d['size'])], 'input_d')

		with tf.variable_scope('g'):
			self.z, self.g = self.createGenerator()
			self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

		with tf.variable_scope('d'):

			self.d, d_real_logits = self.createDiscriminator(self.inputs)
			self._d, d_fake_logits = self.createDiscriminator(self.g, reuse=True)
			#self.d_real, d_real_logits = self.createDiscriminator(self.inputs)
			#self.d_fake, d_fake_logits = self.createDiscriminator(self.g, reuse=True)
			self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)


		with tf.variable_scope('Training'):

			self.labels = tf.placeholder(tf.float32, [None, 1])


			self.loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=self.labels))
			self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.optimize_d = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum).minimize(self.loss_d, var_list=self.d_params)
				self.optimize_g = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum).minimize(self.loss_g, var_list=self.g_params)



	def createDiscriminator(self, inputs, reuse=False):
		with tf.variable_scope('discriminator', reuse=reuse):
			#inputs = x = tf.placeholder(tf.float32, [None, np.sum(self.observation_space_d['size'])], 'input_d')

			if True:
				inputs = tf.split(inputs, self.observation_space_d['size'], axis=1)

				num_filters = 64

				obs = []
				for ob, shape in zip(inputs, self.observation_space_d['shapes']):
					ob_reshaped = tf.reshape(ob, [-1] + list(shape))
					if len(shape) == 3:
						# conv2d
						for i in range(self.d_layers):
							ob_reshaped = tf.layers.conv2d(ob_reshaped, num_filters * (2**i), 3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
							
							if i != 0:
								ob_reshaped = tf.layers.batch_normalization(ob_reshaped, epsilon=1e-5, momentum=0.9)

							ob_reshaped = lrelu(ob_reshaped)

					elif len(shape) == 4:
						for i in range(2):
							ob_reshaped = tf.layers.conv3d(ob_reshaped, 12, 3, activation=tf.nn.relu, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
							ob_reshaped = tf.layers.max_pooling3d(ob_reshaped, 2, 2)
						# conv3d

					obs.append(flatten(ob_reshaped))

				obs = tf.concat(obs, axis=1)
				obs = tf.layers.dense(obs, 128, activation=lrelu, kernel_initializer=tf.contrib.layers.xavier_initializer())
				logits = tf.layers.dense(obs, 1)
				out = tf.nn.sigmoid(logits)

			else:
				obs = tf.layers.dense(inputs, 128, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
				logits = tf.layers.dense(obs, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())
				out = tf.nn.sigmoid(logits)

			return out, logits


	def createGenerator(self, reuse=False, train=True):

		with tf.variable_scope('generator', reuse=reuse):
			inputs = obs = tf.placeholder(tf.float32, [None, np.sum(self.observation_space_g['size'])], 'input_g')

			concatCondition = False
			if len(self.observation_space_d['size']) > 1:
				concatCondition = True
				condition = obs[:, -self.observation_space_d['size'][1]:]
				yb = tf.reshape(condition, (-1, 1, 1, self.observation_space_d['size'][1]))


			s_h, s_w = self.output_space[:2]

			hw = [(s_h, s_w, 0.5)]
			for i in range(self.g_layers):
				h, w, n = hw[-1]
				s_h2, s_w2 = conv_out_size_same(w, 2), conv_out_size_same(w, 2)
				hw.append((s_h2, s_w2, int(n * 2)))

			hw.pop(0)
			hw.reverse()

			#condition = obs[:, -10:]

			num_filters = 64

			if True:

				for i in range(len(hw)):
					h, w, n = hw[i]

					if i == 0:
						obs = tf.layers.dense(obs, h * w * num_filters * n, kernel_initializer=tf.contrib.layers.xavier_initializer())
						obs = tf.reshape(obs, [-1, h, w, num_filters * n])
						obs = tf.layers.batch_normalization(obs, epsilon=1e-5, momentum=0.9)
						obs = tf.nn.relu(obs)
					else:
						obs = tf.layers.conv2d_transpose(obs, filters=num_filters * n, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
						obs = tf.layers.batch_normalization(obs, epsilon=1e-5, momentum=0.9)
						obs = tf.nn.relu(obs)

				obs = tf.layers.conv2d_transpose(obs, filters=self.output_space[-1], kernel_size=3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
				
				# resize if not the right size
				if obs.shape[1] != self.output_space[0] or obs.shape[2] != self.output_space[1]:
					obs = tf.image.resize_images(obs, size=(self.output_space[0], self.output_space[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

				out = tf.nn.tanh(obs)
				out = tf.reshape(out, [-1, np.prod(self.output_space)])

			else:

				obs = tf.layers.dense(obs, 256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
				obs = tf.layers.dense(obs, np.prod(self.output_space), kernel_initializer=tf.contrib.layers.xavier_initializer())
				out = tf.nn.tanh(obs)

			if concatCondition:
				print('conditioning')
				out = tf.concat([out, condition], axis=1)

			return inputs, out


	def train_d(self, inputs, labels):
		sess = tf.get_default_session()
		#_, loss = sess.run([self.optimize_d, self.loss_d], {self._x: inputs, self.labels: labels})
		_, loss = sess.run([self.optimize_d, self.loss_d], {self.inputs: inputs, self.labels: labels})
		return loss

	def train_g(self, inputs):
		sess = tf.get_default_session()
		_, loss = sess.run([self.optimize_g, self.loss_g], {self.z: inputs})
		return loss

	def generate(self, inputs):
		sess = tf.get_default_session()
		out = sess.run(self.g, {self.z: inputs})
		return out

	def sample(self, inputs):
		sess = tf.get_default_session()
		out = sess.run(self.g, {self.z: inputs})

		if len(self.observation_space_d['size']) > 1:
			out = out[:, :-self.observation_space_d['size'][1]]

		return out

	def predict(self, inputs):
		sess = tf.get_default_session()
		out = sess.run(self.d, {self.inputs: inputs})
		return out

	def losses_g(self, inputs_noise):
		sess = tf.get_default_session()
		#_, loss = sess.run([self.optimize_d, self.loss_d], {self._x: inputs, self.labels: labels})
		return sess.run(self.loss_g, {self.z: inputs_noise})

	def losses_d(self, inputs, labels):
		sess = tf.get_default_session()
		#_, loss = sess.run([self.optimize_d, self.loss_d], {self._x: inputs, self.labels: labels})
		return sess.run(self.loss_d, {self.inputs: inputs, self.labels: labels})






if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--logdir', type=str, default='events/%d' % int(time.time() * 1000), help='Directory where checkpoint and summary is stored')
	parser.add_argument('--datadir', type=str, default='data', help='Directory where mnist data is stored')
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--test', action='store_true')

	args = parser.parse_args()

	logdir = args.logdir

	itemdir = os.path.join(args.datadir, 'clean_items')

	#randomItem = getRandomItem(itemdir, 256)

	#print(randomItem)
	#sys.exit(1)

	#loader = Loader('celebA', 108, 64, '*.jpg', flatten=True, gray=False)
	#loader = Loader('clean_items', 24, 64, '*.png', flatten=True, gray=False)
	mnist = input_data.read_data_sets(args.datadir, one_hot=True)

	rand_size = 100
	num_classes = 10

	useConditions = True

	#output_space = (64, 64, 3)
	#output_space = (64, 64, 4)
	output_space = (28, 28, 1)

	#observation_space_d = {'shapes': [(24, 24, 3), (num_classes,)], 'size': [28 * 28, num_classes]}
	#observation_space_g = {'shapes': [(rand_size + num_classes,)], 'size': [rand_size  + num_classes]}

	if useConditions:
		observation_space_d = {'shapes': [output_space, (num_classes,)], 'size': [np.prod(output_space), num_classes]}
		observation_space_g = {'shapes': [(rand_size + num_classes,)], 'size': [rand_size + num_classes]}
	else:
		observation_space_d = {'shapes': [output_space], 'size': [np.prod(output_space)]}
		observation_space_g = {'shapes': [(rand_size,)], 'size': [rand_size]}




	batch_size = 64
	num_batches = 50000


	#gan = DCGAN(observation_space_d, observation_space_g, output_space)
	gan = DCGAN(observation_space_d, observation_space_g, output_space, d_layers=4, g_layers=2)




	summary_writer = tf.summary.FileWriter(logdir)

	# Set up saver
	saver = tf.train.Saver()

	# TODO: set this when loading from a checkpoint
	step_count = 0
	last_checkpoint_time = time.time()
	last_checkpoint_marker = 0
	backup_checkpoint_every_n = 200000


	last_loss_reals = []
	last_loss_fakes = []
	last_loss_wrongs = []
	last_loss_gens = []


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())


		if logdir is not None and os.path.exists(logdir):
			checkpoint_state = tf.train.get_checkpoint_state(logdir)
			if checkpoint_state is not None:
				try:
					saver.restore(sess, checkpoint_state.model_checkpoint_path)
					logger.info('Restoring previous session')
					#step_count = policy.global_step.eval(sess)
				except (tf.errors.NotFoundError):
					logger.info('Could not find checkpoint at %s', checkpoint_state.model_checkpoint_path)


		if args.train:


			sample_noise = np.random.uniform(-1.0, 1.0, (batch_size, rand_size))

			indices = np.zeros((batch_size, num_classes))
			b = np.asarray(range(batch_size)) % num_classes
			indices[np.arange(len(indices)), b] = 1

			# conditions
			if useConditions:
				sample_noise = np.concatenate([sample_noise, indices], axis=1)

			for batch in range(num_batches):



				#batch_xs = loader.next_batch(batch_size)
				#batch_xs = getRandomItem(itemdir, batch_size)
				
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)

				# scale mnist data to -1 to 1
				batch_xs = (batch_xs * 2) - 1

				if useConditions:
					# add conditional
					wrong_labels = np.roll(batch_ys, np.random.choice(num_classes / 2) + 1)
					reals = np.concatenate([batch_xs, batch_ys], axis=1)
					wrongs = np.concatenate([batch_xs, wrong_labels], axis=1)


				#print(batch_xs.shape, batch_ys.shape)

				# create noise vectors z

				if True: # train DCGAN
					noise = np.random.uniform(-1.0, 1.0, (batch_size, rand_size))
					if useConditions:
						# append condition
						noise = np.concatenate([noise, batch_ys], axis=1)

					samples = gan.generate(noise)

					#print(batch_xs.shape, samples.shape)

					if useConditions:
						train_batch = np.concatenate([reals, wrongs, samples], axis=0)
						#train_batch = np.concatenate([reals, wrongs], axis=0)
					else:
						train_batch = np.concatenate([batch_xs, samples], axis=0)


					if useConditions:
						labels = np.ones((batch_size * 3, 1))
						#labels = np.ones((batch_size * 2, 1))
					else:
						labels = np.ones((batch_size * 2, 1))


					labels[batch_size:,:] = 0

					#print(labels)


					d_loss = gan.train_d(train_batch, labels)


					g_loss1 = gan.train_g(noise)
					g_loss2 = gan.train_g(noise)
					#g_loss3 = gan.train_g(noise)

					loss_d = gan.losses_d(train_batch, labels)
					loss_g = gan.losses_g(noise)
					print('Batch: %4d, d_loss: %.8f, g_loss: %.8f' % (batch, loss_d, loss_g))


					if batch % 100 == 0:

						splits = np.array_split(gan.predict(train_batch), 3, axis=0)
						print(np.mean(splits, axis=1))

						#print(gan.predict(train_batch))

						samples = gan.sample(sample_noise)

						samples = (samples[0:64]+1) / 2.0
						reals = (batch_xs[0:64]+1) / 2.0

						print(np.min(samples), np.max(samples), np.min(reals), np.max(reals))

						fig = plot(samples, *output_space)
						plt.savefig('{}/{}.png'.format(logdir, str(batch).zfill(3)), bbox_inches='tight')
						plt.close(fig)

						fig = plot(reals, *output_space)
						plt.savefig('{}/{}_real.png'.format(logdir, str(batch).zfill(3)), bbox_inches='tight')
						plt.close(fig)



				global_step = batch

				if time.time() - last_checkpoint_time > 60:
					saver.save(sess, os.path.join(logdir, 'model.ckpt'), global_step)
					last_checkpoint_time = time.time()

					# backup checkpoints every 10k steps
					if global_step > last_checkpoint_marker:
						last_checkpoint_marker += backup_checkpoint_every_n
						backup_dir = os.path.join(logdir, 'backup')
						if not os.path.isdir(backup_dir):
							os.mkdir(backup_dir)

						for root, dirs, files in os.walk(logdir):
							for file in files:
								orig_file = os.path.join(root, file)
								backup_file = os.path.join(backup_dir, file)
								if str(global_step) in file and orig_file != backup_file:
									shutil.copy(orig_file, backup_file)


		elif args.test:
			pass
