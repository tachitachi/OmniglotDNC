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


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

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


def getRandomItem(itemdir, batch_size):
	for root, dirs, files in os.walk(itemdir):


		arrs = []

		randomFiles = np.random.choice(files, batch_size, replace=False)

		for file in randomFiles:
			filepath = os.path.join(root, file)

			arr = imread(filepath)
			arr = rgb2gray(arr)

			if np.random.random() < 0.5:
				arr = np.flip(arr, 0)
			if np.random.random() < 0.5:
				arr = np.flip(arr, 1)

			#print(arr.shape)
			#arr = arr[:,:,:3]
			arr = np.reshape(arr, [-1])
			arrs.append(arr)

		return np.array(arrs)





class DCGAN(object):
	def __init__(self, observation_space_d, observation_space_g, output_space, batch_size=32):
		self.observation_space_d = observation_space_d
		self.observation_space_g = observation_space_g
		self.output_space = output_space

		self.batch_size = batch_size


		self.inputs = tf.placeholder(tf.float32, [None, np.sum(self.observation_space_d['size'])], 'input_d')
		#self.inputs = tf.placeholder(tf.float32, [None] + list(self.observation_space_d['shapes'][0]), 'input_d')

		with tf.variable_scope('g'):
			#self.g_bn0 = batch_norm(name='g_bn0')
			#self.g_bn1 = batch_norm(name='g_bn1')
			#self.g_bn2 = batch_norm(name='g_bn2')
			#self.g_bn3 = batch_norm(name='g_bn3')

			self.z, self.g = self.createGenerator()

			self._z, self._g = self.createGenerator(reuse=True, train=False)

			self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

		with tf.variable_scope('d'):


			#self.d_bn1 = batch_norm(name='d_bn1')
			#self.d_bn2 = batch_norm(name='d_bn2')
			#self.d_bn3 = batch_norm(name='d_bn3')

			self.d_real, d_real_logits = self.createDiscriminator(self.inputs)
			self.d_fake, d_fake_logits = self.createDiscriminator(self.g, reuse=True)
			self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)


		with tf.variable_scope('Training'):

			self.labels = tf.placeholder(tf.float32, [None, 1])


			#self.loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logits, labels=self.labels))
			#self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logits, labels=tf.ones_like(self.d_logits)))

			#self.loss_d = -tf.reduce_mean(tf.log(self.d_real) + tf.log(1 - self.d_fake))
			#self.loss_g = -tf.reduce_mean(tf.log(self.d_fake))

			D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits)))
			D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))
			self.loss_d = D_loss_real + D_loss_fake
			self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.optimize_d = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.loss_d, var_list=self.d_params)
				self.optimize_g = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.loss_g, var_list=self.g_params)



	def createDiscriminator(self, inputs, reuse=False):
		with tf.variable_scope('discriminator', reuse=reuse):
			#inputs = x = tf.placeholder(tf.float32, [None, np.sum(self.observation_space_d['size'])], 'input_d')

			if False:
				self.df_dim = 64
				image = tf.reshape(inputs, [-1] + list(self.observation_space_d['shapes'][0]))
				#image = inputs
				h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
				h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
				h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
				h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
				print(h3)
				#h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
				h4 = linear(flatten(h3), 1, 'd_h4_lin')

				logits = h4
				out = tf.nn.sigmoid(h4)

			elif True:
				inputs = tf.split(inputs, self.observation_space_d['size'], axis=1)

				num_filters = 64

				obs = []
				for ob, shape in zip(inputs, self.observation_space_d['shapes']):
					ob_reshaped = tf.reshape(ob, [-1] + list(shape))
					if len(shape) == 3:
						# conv2d

						if True:
							ob_reshaped = tf.layers.conv2d(ob_reshaped, num_filters, 3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
							ob_reshaped = lrelu(ob_reshaped)

							ob_reshaped = tf.layers.conv2d(ob_reshaped, num_filters * 2, 3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
							#ob_reshaped = self.d_bn1(ob_reshaped)
							ob_reshaped = tf.layers.batch_normalization(ob_reshaped)
							ob_reshaped = lrelu(ob_reshaped)

							ob_reshaped = tf.layers.conv2d(ob_reshaped, num_filters * 4, 3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
							#ob_reshaped = self.d_bn2(ob_reshaped)
							ob_reshaped = tf.layers.batch_normalization(ob_reshaped)
							ob_reshaped = lrelu(ob_reshaped)

							ob_reshaped = tf.layers.conv2d(ob_reshaped, num_filters * 8, 3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
							#ob_reshaped = self.d_bn3(ob_reshaped)
							ob_reshaped = tf.layers.batch_normalization(ob_reshaped)
							ob_reshaped = lrelu(ob_reshaped)

						else:
							for i in range(3):
								ob_reshaped = tf.layers.conv2d(ob_reshaped, num_filters * (2**i), 3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
								
								#if ob_reshaped.shape[1] > 1 and ob_reshaped.shape[2] > 1:
								#	ob_reshaped = tf.layers.max_pooling2d(ob_reshaped, 2, 2)
								ob_reshaped = tf.layers.batch_normalization(ob_reshaped)
								#ob_reshaped = tf.nn.relu(ob_reshaped)
								ob_reshaped = lrelu(ob_reshaped)

					elif len(shape) == 4:
						for i in range(2):
							ob_reshaped = tf.layers.conv3d(ob_reshaped, 12, 3, activation=tf.nn.relu, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
							ob_reshaped = tf.layers.max_pooling3d(ob_reshaped, 2, 2)
						# conv3d

					obs.append(flatten(ob_reshaped))

				obs = tf.concat(obs, axis=1)

				#obs = tf.layers.dense(obs, 128)

				logits = tf.layers.dense(obs, 1)

				out = tf.nn.sigmoid(logits)

			else:
				obs = inputs
				obs = tf.layers.dense(inputs, 256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
				#obs = tf.layers.dropout(obs, rate=0.4)
				obs = tf.layers.dense(obs, 128, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
				#obs = tf.layers.dropout(obs, rate=0.4)
				logits = tf.layers.dense(obs, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())

				out = tf.nn.sigmoid(logits)


			return out, logits


	def createGenerator(self, reuse=False, train=True):

		with tf.variable_scope('generator', reuse=reuse):
			inputs = obs = tf.placeholder(tf.float32, [None, np.sum(self.observation_space_g['size'])], 'input_g')


			s_h, s_w = self.output_space[:2]
			s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
			s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
			s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
			s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

			print(s_h, s_w)
			print(s_h2, s_w2)
			print(s_h4, s_w4)
			print(s_h8, s_w8)
			print(s_h16, s_w16)


			#condition = obs[:, -10:]

			num_filters = 64
			self.gf_dim = 64

			if True:
				# / 16
				obs = tf.layers.dense(obs, s_h16 * s_w16 * num_filters * 8, kernel_initializer=tf.contrib.layers.xavier_initializer())
				obs = tf.reshape(obs, [-1, s_h16, s_w16, num_filters * 8])
				obs = tf.layers.batch_normalization(obs)
				obs = tf.nn.relu(obs)

				# / 8
				obs = tf.layers.conv2d_transpose(obs, filters=num_filters * 4, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
				obs = tf.layers.batch_normalization(obs)
				obs = tf.nn.relu(obs)

				# / 4
				obs = tf.layers.conv2d_transpose(obs, filters=num_filters * 2, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
				obs = tf.layers.batch_normalization(obs)
				obs = tf.nn.relu(obs)

				# / 2
				obs = tf.layers.conv2d_transpose(obs, filters=num_filters * 1, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
				obs = tf.layers.batch_normalization(obs)
				obs = tf.nn.relu(obs)

				# / 1
				obs = tf.layers.conv2d_transpose(obs, filters=self.output_space[-1], kernel_size=3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
				
				out = tf.nn.tanh(obs)
				out = tf.reshape(out, [-1, np.prod(self.output_space)])


			elif False:


				z = inputs

				# project `z` and reshape
				z_, h0_w, h0_b = linear(
					z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

				h0 = tf.reshape(
					z_, [-1, s_h16, s_w16, self.gf_dim * 8])
				h0 = tf.nn.relu(self.g_bn0(h0))

				h1, h1_w, h1_b = deconv2d(
					h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
				h1 = tf.nn.relu(self.g_bn1(h1, train=train))

				h2, h2_w, h2_b = deconv2d(
					h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
				h2 = tf.nn.relu(self.g_bn2(h2, train=train))

				h3, h3_w, h3_b = deconv2d(
					h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
				h3 = tf.nn.relu(self.g_bn3(h3, train=train))

				h4, self.h4_w, self.h4_b = deconv2d(
					h3, [self.batch_size, s_h, s_w, self.output_space[-1]], name='g_h4', with_w=True)

				#return tf.nn.tanh(h4)
				out = tf.nn.tanh(h4)
				#out = tf.nn.sigmoid(h4)

				#out = tf.reshape(out, [-1, np.prod(self.output_space)])

			else:

				obs = tf.layers.dense(obs, 256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
				obs = tf.layers.dense(obs, np.prod(self.output_space), kernel_initializer=tf.contrib.layers.xavier_initializer())
				out = tf.nn.sigmoid(obs)

			#out = tf.concat([out, condition], axis=1)

			return inputs, out


	def train_d(self, inputs_real, inputs_noise):
		sess = tf.get_default_session()
		#_, loss = sess.run([self.optimize_d, self.loss_d], {self._x: inputs, self.labels: labels})
		_, loss = sess.run([self.optimize_d, self.loss_d], {self.inputs: inputs_real, self.z: inputs_noise})
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
		out = sess.run(self._g, {self._z: inputs})
		return out

	def predict(self, inputs):
		sess = tf.get_default_session()
		out = sess.run(self.d, {self.inputs: inputs})
		return out

	def losses(self, inputs_real, inputs_noise):
		sess = tf.get_default_session()
		#_, loss = sess.run([self.optimize_d, self.loss_d], {self._x: inputs, self.labels: labels})
		loss_d, loss_g = sess.run([self.loss_d, self.loss_g], {self.inputs: inputs_real, self.z: inputs_noise})
		return loss_d, loss_g






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

	loader = Loader('celebA', 108, 64, '*.jpg', flatten=True, gray=False)
	#loader = Loader('clean_items', 24, 24, '*.png', flatten=False, gray=False)
	#mnist = input_data.read_data_sets(args.datadir, one_hot=True)

	rand_size = 100
	num_classes = 10

	output_space = (64, 64, 3)
	#output_space = (24, 24, 4)
	#output_space = (28, 28, 1)

	#observation_space_d = {'shapes': [(24, 24, 3), (num_classes,)], 'size': [28 * 28, num_classes]}
	#observation_space_g = {'shapes': [(rand_size + num_classes,)], 'size': [rand_size  + num_classes]}
	observation_space_d = {'shapes': [output_space], 'size': [np.prod(output_space)]}
	observation_space_g = {'shapes': [(rand_size,)], 'size': [rand_size]}



	batch_size = 64
	num_batches = 50000


	#gan = DCGAN(observation_space_d, observation_space_g, output_space)
	gan = DCGAN(observation_space_d, observation_space_g, output_space, batch_size=batch_size)





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

			for batch in range(num_batches):



				#batch_xs = getRandomItem(itemdir, batch_size)
				#batch_xs, _ = mnist.train.next_batch(batch_size)
				batch_xs = loader.next_batch(batch_size)

				#print(batch_xs.shape, batch_ys.shape)

				# create noise vectors z

				if True:

					if True: # train DCGAN
						noise = np.random.uniform(-1.0, 1.0, (batch_size, rand_size))
						#noise = np.random.normal(0, 1.0, (batch_size, rand_size))
						d_loss = gan.train_d(batch_xs, noise)

						#print(batch_xs)

						#noise = np.random.uniform(-1.0, 1.0, (batch_size, rand_size))
						#noise = np.random.normal(0, 1.0, (batch_size, rand_size))
						g_loss1 = gan.train_g(noise)

						#noise = np.random.uniform(0, 1.0, (batch_size, rand_size))
						#noise = np.random.normal(0, 1.0, (batch_size, rand_size))
						g_loss2 = gan.train_g(noise)

						loss_d, loss_g = gan.losses(batch_xs, noise)
						print('Batch: %4d, d_loss: %.8f, g_loss: %.8f' % (batch, loss_d, loss_g))


						if batch % 100 == 0:

							#print(d_loss, g_loss1, g_loss2)

							#noise = np.random.normal(0, 1.0, (batch_size, rand_size))
							samples = gan.sample(sample_noise)

							#print(samples)

							samples = (samples[0:64]+1) / 2.0
							reals = (batch_xs[0:64]+1) / 2.0

							print(np.min(samples), np.max(samples), np.min(reals), np.max(reals))

							#samples = (samples * 255).astype(np.int32)
							#print(np.max(samples, 1), np.min(samples, 1))

							fig = plot(samples, *output_space)
							plt.savefig('{}/{}.png'.format(logdir, str(batch).zfill(3)), bbox_inches='tight')
							plt.close(fig)


							fig = plot(reals, *output_space)
							plt.savefig('{}/{}_real.png'.format(logdir, str(batch).zfill(3)), bbox_inches='tight')
							plt.close(fig)

					else: # train CondGAN
						noise = np.random.uniform(-1.0, 1.0, (batch_size, rand_size))
						cond_noise = np.concatenate([noise, batch_ys], axis=1)
						fake_samples = gan.generate(cond_noise)

						labels = np.ones((batch_size * 3, 1))
						labels[batch_size:,:] = 0

						wrong_labels = np.roll(batch_ys, np.random.choice(num_classes / 2) + 1)

						#print(batch_ys[:3,:])
						#print(wrong_labels[:3,:])
						#print

						reals = np.concatenate([batch_xs, batch_ys], axis=1)
						wrongs = np.concatenate([batch_xs, wrong_labels], axis=1)
						#fakes = np.concatenate([fake_samples, batch_ys], axis=1)
						fakes = fake_samples

						#print(reals.shape, fakes.shape)

						all_samples = np.concatenate([reals, wrongs, fakes], axis=0)

						d_loss = gan.train_d(all_samples, labels)

						#noise = np.random.uniform(-1.0, 1.0, (batch_size, rand_size))
						g_loss = gan.train_g(cond_noise)


						if batch % 1000 == 0:

							indices = np.zeros((16, num_classes))
							b = np.asarray(range(16)) % num_classes
							indices[np.arange(len(indices)), b] = 1


							noise = np.random.uniform(-1.0, 1.0, (16, rand_size))
							cond_noise = np.concatenate([noise, indices], axis=1)
							samples = gan.generate(cond_noise)

							samples = samples[:, :28*28]
							samples = (samples - 255) * -1

							fig = plot(samples)
							plt.savefig('{}/{}.png'.format(logdir, str(batch).zfill(3)), bbox_inches='tight')
							plt.close(fig)


					#if batch % 100 == 0:
					#	print(d_loss, g_loss)





				if False:

					_, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: batch_xs, Z: sample_Z(batch_size, rand_size)})
					_, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, rand_size)})


					if batch % 100 == 0:
						print(D_loss_curr, G_loss_curr)

					
					if batch % 1000 == 0:

						samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, rand_size)})

						#noise = np.random.uniform(-1.0, 1.0, (16, rand_size))
						#samples = gan.generate(noise)

						fig = plot(samples)
						plt.savefig('{}/{}.png'.format(logdir, str(batch).zfill(3)), bbox_inches='tight')
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
