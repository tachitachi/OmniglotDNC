import os

from scipy.misc import imresize, imread
from skimage.color import rgb2gray
from glob import glob

import numpy as np

class Loader(object):
	def __init__(self, dataset, input_size, output_size, fname_format='*.jpg', flatten=False, gray=False):
		self.dataset = dataset
		self.fname_format = fname_format

		self.input_size = input_size
		self.output_size = output_size
		self.flatten = flatten
		self.gray = gray


		self.current = 0


		self.files = glob(os.path.join('data', self.dataset, self.fname_format))

		self.data = {}

	@property
	def length(self):
		return len(self.files)

	def get(self, filepath):
		if filepath not in self.data:
			arr = imread(filepath)

			# center crop image to input_size
			h, w = arr.shape[:2]
			i = int(round((h - self.input_size) / 2.0))
			j = int(round((w - self.input_size) / 2.0))

			arr = arr[j:j+self.input_size, i:i+self.input_size]

			# resize to output_size
			if self.input_size != self.output_size:
				arr = imresize(arr, [self.output_size, self.output_size])

			if self.gray:
				arr = rgb2gray(arr)

			if self.flatten:
				arr = np.reshape(arr, [-1])

			self.data[filepath] = arr

		return self.data[filepath]

	def next_batch(self, batch_size):
		if self.current + batch_size <= self.length:
			batch = [self.get(filepath) for filepath in self.files[self.current:self.current+batch_size]]
			self.current += batch_size
			return np.asarray(batch)
		else:

			spillover = (self.current + batch_size) % self.length

			end = [self.get(filepath) for filepath in self.files[self.current:]]
			front = [self.get(filepath) for filepath in self.files[:spillover]]

			self.current = spillover

			return np.asarray(end + front)


