import os
import shutil

import hashlib

from scipy.misc import imread

# BUF_SIZE is totally arbitrary, change for your app!
BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

unique = set()

for root, dirs, files in os.walk('data/item'):

	for file in files:
		filepath = os.path.join(root, file)

		im = imread(filepath)
		if im.shape != (24, 24, 4):
			continue

		md5 = hashlib.md5()
		sha1 = hashlib.sha1()

		with open(filepath, 'rb') as f:
		    while True:
		        data = f.read(BUF_SIZE)
		        if not data:
		            break
		        md5.update(data)
		        sha1.update(data)

		h = md5.hexdigest()

		if h not in unique:
			unique.add(h)
			if not os.path.isdir('data/clean_items'):
				os.mkdir('data/clean_items')
			shutil.copy(filepath, os.path.join('data', 'clean_items', file))