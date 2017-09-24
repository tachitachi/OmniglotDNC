import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import shutil
import argparse
import time
import cPickle as pickle
import sys

import logging
#import cProfile as profile

from tensorflow.examples.tutorials.mnist import input_data

from scipy.misc import imresize, imsave, imread
from skimage.color import rgb2gray

from ops import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_loader import Loader

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # set to INFO if you want fewer messages

def plot(samples):
	fig = plt.figure(figsize=(4, 4))
	gs = gridspec.GridSpec(4, 4)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(24, 24), cmap='gray')
		#plt.imshow(sample.reshape(24, 24, 3))

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
	def __init__(self, observation_space_d, observation_space_g, output_space):
		self.observation_space_d = observation_space_d
		self.observation_space_g = observation_space_g
		self.output_space = output_space


		self.inputs = tf.placeholder(tf.float32, [None, np.sum(self.observation_space_d['size'])], 'input_d')

		with tf.variable_scope('g'):
			self.z, self.g = self.createGenerator()
			self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

		with tf.variable_scope('d'):
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
				self.optimize_d = tf.train.AdamOptimizer(5e-4).minimize(self.loss_d, var_list=self.d_params)
				self.optimize_g = tf.train.AdamOptimizer(1e-3).minimize(self.loss_g, var_list=self.g_params)



	def createDiscriminator(self, inputs, reuse=False):
		with tf.variable_scope('discriminator', reuse=reuse):
			#inputs = x = tf.placeholder(tf.float32, [None, np.sum(self.observation_space_d['size'])], 'input_d')


			if True:
				inputs = tf.split(inputs, self.observation_space_d['size'], axis=1)

				obs = []
				for ob, shape in zip(inputs, self.observation_space_d['shapes']):
					ob_reshaped = tf.reshape(ob, [-1] + list(shape))
					if len(shape) == 3:
						# conv2d
						for i in range(3):
							ob_reshaped = tf.layers.conv2d(ob_reshaped, 2**(i+4), 3, activation=tf.nn.relu, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
							ob_reshaped = tf.layers.max_pooling2d(ob_reshaped, 2, 2)
							ob_reshaped = tf.layers.batch_normalization(ob_reshaped)

						pass
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


	def createGenerator(self):

		with tf.variable_scope('generator'):
			inputs = obs = tf.placeholder(tf.float32, [None, np.sum(self.observation_space_g['size'])], 'input_g')

			#condition = obs[:, -10:]

			if False:

				obs = tf.layers.dense(obs, 3 * 3 * 1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

				obs = tf.reshape(obs, [-1, 3, 3, 1])

				obs = tf.layers.conv2d_transpose(obs, filters=128, kernel_size=3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
				#obs = tf.layers.batch_normalization(obs)
				obs = tf.nn.relu(obs)
				obs = tf.image.resize_images(obs, size=(6, 6))
				#obs = tf.image.resize_images(obs, size=(7, 7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

				obs = tf.layers.conv2d_transpose(obs, filters=64, kernel_size=3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
				#obs = tf.layers.batch_normalization(obs)
				obs = tf.nn.relu(obs)
				obs = tf.image.resize_images(obs, size=(12, 12))
				#obs = tf.image.resize_images(obs, size=(14, 14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

				obs = tf.layers.conv2d_transpose(obs, filters=32, kernel_size=3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
				#obs = tf.layers.batch_normalization(obs)
				obs = tf.nn.relu(obs)
				obs = tf.image.resize_images(obs, size=(24, 24))
				#obs = tf.image.resize_images(obs, size=(28, 28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


				obs = tf.layers.conv2d_transpose(obs, filters=self.output_shape[-1], kernel_size=1, padding='same', activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

				out = tf.nn.sigmoid(obs)

				out = tf.reshape(out, [-1, np.prod(self.output_space)])

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

	def predict(self, inputs):
		sess = tf.get_default_session()
		out = sess.run(self.d, {self.inputs: inputs})
		return out























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

	loader = Loader('clean_items', 24, 24, '*.png', flatten=True, gray=True)


	rand_size = 100
	num_classes = 10

	output_space = (24, 24, 1)

	#observation_space_d = {'shapes': [(24, 24, 3), (num_classes,)], 'size': [28 * 28, num_classes]}
	#observation_space_g = {'shapes': [(rand_size + num_classes,)], 'size': [rand_size  + num_classes]}
	observation_space_d = {'shapes': [output_space], 'size': [np.prod(output_space)]}
	observation_space_g = {'shapes': [(rand_size,)], 'size': [rand_size]}





	#gan = DCGAN(observation_space_d, observation_space_g, output_space)
	gan = DCGAN(observation_space_d, observation_space_g, output_space)



	batch_size = 256
	num_batches = 50000


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

			for batch in range(num_batches):



				#batch_xs = getRandomItem(itemdir, batch_size)
				batch_xs = loader.next_batch(batch_size)

				#print(batch_xs.shape, batch_ys.shape)

				# create noise vectors z

				if True:

					if True: # train DCGAN
						#noise = np.random.uniform(-1.0, 1.0, (batch_size, rand_size))
						noise = np.random.normal(0, 1.0, (batch_size, rand_size))
						d_loss = gan.train_d(batch_xs, noise)

						#noise = np.random.uniform(-1.0, 1.0, (batch_size, rand_size))
						noise = np.random.normal(0, 1.0, (batch_size, rand_size))
						g_loss = gan.train_g(noise)


						if batch % 200 == 0:

							#noise = np.random.uniform(-1.0, 1.0, (16, rand_size))
							noise = np.random.normal(0, 1.0, (16, rand_size))
							samples = gan.generate(noise)

							#samples = (samples * 255).astype(np.int32)
							#print(np.max(samples, 1), np.min(samples, 1))

							fig = plot(samples)
							plt.savefig('{}/{}.png'.format(logdir, str(batch).zfill(3)), bbox_inches='tight')
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


					if batch % 100 == 0:
						print(d_loss, g_loss)





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
