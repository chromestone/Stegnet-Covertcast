"""
TODO
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

		the_rng = np.random.default_rng(seed)
		self.random_stuff = the_rng.integers(2, size=length - 1 + self.bits_per_img,
												dtype=np.uint8)

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

		return img
