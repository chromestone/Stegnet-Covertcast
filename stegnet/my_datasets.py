"""
This module contains custom defined PyTorch datasets.
"""

import numpy as np

from torch.utils.data import Dataset

class RandomDataset(Dataset):

	def __init__(self, length, seed, img_size, six_bit_res, transform=None):

		assert length >= 1
		h, w = img_size
		h_res, w_res = six_bit_res
		assert h % h_res == 0 and w % w_res == 0

		self.length = length
		self.img_size = img_size
		self.six_bit_res = six_bit_res
		self.bits_per_img = (h // h_res) * (w // w_res) * 6
		self.transform = transform

		the_rng = np.random.default_rng(seed)
		self.random_stuff = the_rng.integers(2, size=length - 1 + self.bits_per_img,
												dtype=np.uint8)
		self.test = False

	def __len__(self):

		return self.length

	def __getitem__(self, idx):

		h, w = self.img_size
		h_res, w_res = self.six_bit_res
		stuff = np.reshape(self.random_stuff[idx : idx + self.bits_per_img],
							(h // h_res, w // w_res, 6))

		rgb_stuff = np.empty((h // h_res, w // w_res, 3), dtype=np.uint8)
		rgb_stuff[:, :, 0] = 160 * stuff[:, :, 0] + 32 * stuff[:, :, 1]
		rgb_stuff[:, :, 1] = 160 * stuff[:, :, 2] + 32 * stuff[:, :, 3]
		rgb_stuff[:, :, 2] = 160 * stuff[:, :, 4] + 32 * stuff[:, :, 5]

		img = np.empty((h, w, 3), dtype=np.uint8)
		for i in range(h_res):

			for j in range(w_res):

				img[i::h_res, j::w_res] = rgb_stuff

		if self.transform:

			img = self.transform(img)

		if not self.test:

			return img

		else:

			return img, stuff

class BitArrayDataset(Dataset):

	def __init__(self, bit_array, stride, img_size, six_bit_res, transform=None):
		"""
		bit_array is any numpy array of 0's and 1's
		"""

		h, w = img_size
		h_res, w_res = six_bit_res
		assert h % h_res == 0 and w % w_res == 0

		self.img_size = img_size
		self.six_bit_res = six_bit_res
		self.bits_per_img = (h // h_res) * (w // w_res) * 6
		self.transform = transform

		assert bit_array.shape[0] >= self.bits_per_img
		self.bit_array = bit_array
		self.length = bit_array.shape[0] + 1 - self.bits_per_img
		assert stride <= self.length
		self.stride = stride
		self.test = False

	def __len__(self):

		return self.length // self.stride

	def __getitem__(self, idx):

		idx *= self.stride

		h, w = self.img_size
		h_res, w_res = self.six_bit_res
		stuff = np.reshape(self.bit_array[idx : idx + self.bits_per_img],
							(h // h_res, w // w_res, 6))

		rgb_stuff = np.empty((h // h_res, w // w_res, 3), dtype=np.uint8)
		rgb_stuff[:, :, 0] = 160 * stuff[:, :, 0] + 32 * stuff[:, :, 1]
		rgb_stuff[:, :, 1] = 160 * stuff[:, :, 2] + 32 * stuff[:, :, 3]
		rgb_stuff[:, :, 2] = 160 * stuff[:, :, 4] + 32 * stuff[:, :, 5]

		img = np.empty((h, w, 3), dtype=np.uint8)
		for i in range(h_res):

			for j in range(w_res):

				img[i::h_res, j::w_res] = rgb_stuff

		if self.transform:

			img = self.transform(img)

		if not self.test:

			return img

		else:

			return img, stuff

class TransmitDataset(BitArrayDataset):
	"""
	Adds sequence variable and random null bytes at the end.
	"""

	def __init__(self, bit_array, stride, img_size, six_bit_res, seed, transform=None):
		"""
		bit_array is any numpy array of 0's and 1's
		"""

		super().__init__(bit_array, stride, img_size, six_bit_res, transform)
		if isinstance(seed, int):

			self.rng = np.random.default_rng(seed) if self.bits_per_img // 32 else None

			if self.rng is None:

				print(f'Warning {self.bits_per_img} not divisible by 32 = 4 * 8.\n'
						'Zero out not used.')

		else:

			self.rng = None

	def __getitem__(self, idx):

		idx *= self.stride

		h, w = self.img_size
		h_res, w_res = self.six_bit_res
		stuff = np.empty((self.bits_per_img, ), dtype=np.uint8)
		stuff[:] = self.bit_array[idx : idx + self.bits_per_img]

		for i, the_bit in enumerate(format(idx % 256, '08b')):

			stuff[i] = the_bit

		if self.rng is not None:

			zero_out = self.rng.integers(0, self.bits_per_img // 32)
			if zero_out > 0:

				stuff[-zero_out * 8 :] = 0

		stuff = np.reshape(stuff, (h // h_res, w // w_res, 6))

		rgb_stuff = np.empty((h // h_res, w // w_res, 3), dtype=np.uint8)
		rgb_stuff[:, :, 0] = 160 * stuff[:, :, 0] + 32 * stuff[:, :, 1]
		rgb_stuff[:, :, 1] = 160 * stuff[:, :, 2] + 32 * stuff[:, :, 3]
		rgb_stuff[:, :, 2] = 160 * stuff[:, :, 4] + 32 * stuff[:, :, 5]

		img = np.empty((h, w, 3), dtype=np.uint8)
		for i in range(h_res):

			for j in range(w_res):

				img[i::h_res, j::w_res] = rgb_stuff

		if self.transform:

			img = self.transform(img)

		if not self.test:

			return img

		else:

			return img, stuff

def dataset_from_text(filepath, stride, img_size, six_bit_res, transform=None):

	with open(filepath, 'r', encoding='utf-8') as fp:

		line_list = fp.readlines()

	def _bit_iter():

		for line in line_list:

			for byte_str in bytearray(line, 'utf-8'):

				for the_bit in format(byte_str, '08b'):

					yield the_bit

	bit_array = np.fromiter(_bit_iter(), dtype=np.uint8)
	return BitArrayDataset(bit_array, stride, img_size, six_bit_res, transform)

def binary_bit_iterator(binary_content):

	for byte_str in binary_content:

		for the_bit in format(byte_str, '08b'):

			yield the_bit
	del binary_content

def dataset_from_binary(filepath, stride, img_size, six_bit_res, random, transform=None):

	with open(filepath, 'rb') as fp:

		binary_content = fp.read()

	bit_array = np.fromiter(binary_bit_iterator(binary_content), dtype=np.uint8)
	del binary_content

	h, w = img_size
	h_res, w_res = six_bit_res
	assert h % h_res == 0 and w % w_res == 0
	bits_per_img = (h // h_res) * (w // w_res) * 6
	length = bit_array.shape[0] + 1 - bits_per_img
	if bit_array.shape[0] < bits_per_img or stride > length:

		return None

	return TransmitDataset(bit_array, stride, img_size, six_bit_res,
							123456789 if random else None, transform)
