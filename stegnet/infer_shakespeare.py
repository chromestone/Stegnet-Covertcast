"""
TODO
"""

import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision import datasets

import cv2
import numpy as np
from tqdm import tqdm

from model import Stegnet

from my_datasets import dataset_from_text

BATCH_SIZE = 50
SIX_BIT_RES = (2, 2)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

assert len(sys.argv) > 1, 'Specify path to a .pth file.'
assert len(sys.argv) > 2, 'Specify input directory.'
assert len(sys.argv) > 3, 'Specify output directory.'

WEIGHT_PATH = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH = sys.argv[3]

data_transform = T.Compose([
	T.ToTensor(),
	T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mean_arr = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std_arr = np.array([0.229, 0.224, 0.225], dtype=np.float32)

test_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, 'test'), transform=data_transform)
assert len(test_dataset) % (BATCH_SIZE) == 0, len(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
								shuffle=False, drop_last=True)

text_data_transform = T.Compose([
	T.ToTensor(),
	# possible values are 0, 32, 160, 192 (divided by 255 I think)
	# so midpoint that we want to be 0 is 96. 96/255 is about 0.376.
	T.Normalize(mean=[0.376, 0.376, 0.376], std=[0.376, 0.376, 0.376])
])

secret_mean_arr = torch.as_tensor(np.array([[[0.376]], [[0.376]], [[0.376]]], dtype=np.float32), device=DEVICE)
secret_std_arr = torch.as_tensor(np.array([[[0.376]], [[0.376]], [[0.376]]], dtype=np.float32), device=DEVICE)

test_text_dataset = dataset_from_text(os.path.join(self.data_path, 'shakespeare_test.txt'),
												43, (64, 64), SIX_BIT_RES,
												transform=text_data_transform)
test_text_dataset.test = True
test_text_dataloader = DataLoader(test_text_dataset, batch_size=BATCH_SIZE,
								shuffle=False, drop_last=True)

def bit_level_accuracy(outputs, ground_truth):

	transformed_outputs = ((outputs * secret_std_arr) + secret_mean_arr) * 255
	# basically unexpand image
	decoded = F.avg_pool2d(transformed_outputs, SIX_BIT_RES).detach().cpu().numpy()
	decoded = np.transpose(decoded, (0, 2, 3, 1))
	_, height, width, _ = decoded.shape

	decoded_bits = np.empty((BATCH_SIZE, height, width, 3, 2), dtype=np.uint8)
	decoded_bits[decoded <= 16] = [0, 0]
	decoded_bits[(decoded > 16) & (decoded <= 96)] = [0, 1]
	decoded_bits[(decoded > 96) & (decoded <= 176)] = [1, 0]
	decoded_bits[decoded > 176] = [1, 1]
	decoded_bits = np.reshape(decoded_bits, (BATCH_SIZE, height, width, 6))
	return np.mean(decoded_bits == ground_truth)

encoder = Stegnet(6).to(DEVICE)
decoder = Stegnet(3).to(DEVICE)

checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')
encoder.load_state_dict(checkpoint['encoder_dict'])
decoder.load_state_dict(checkpoint['decoder_dict'])

cover_mse = 0.0
secret_mse = 0.0
secret_accuracy = 0
quantized_secret_mse = 0.0
quantized_secret_accuracy = 0
with torch.no_grad():

	for i, (data, text_data) in tqdm(enumerate(zip(test_dataloader, test_text_dataloader),
										start=1)):

		covers, _ = data
		secrets, ground_truth = text_data

		covers = covers.to(DEVICE)
		secrets = secrets.to(DEVICE)
		ground_truth = ground_truth.detach().cpu().numpy()

		embeds = encoder(torch.cat([covers, secrets], dim=1))
		outputs = decoder(embeds)

		cover_mse += F.mse_loss(embeds, covers).item()
		secret_mse += F.mse_loss(outputs, secrets).item()
		secret_accuracy += bit_level_accuracy(outputs, ground_truth)

		np_embeds = np.transpose(embeds.detach().cpu().numpy(), (0, 2, 3, 1))
		np_embeds = (np_embeds * std_arr) + mean_arr
		np_embeds = np_embeds * 255

		# this is really quantized
		quantized_embeds = np.clip(np_embeds, 0, 255).astype(np.uint8)

		# these are floats, only using quantized as variable name
		quantized_tensor = quantized_embeds.astype(np.float32) / 255
		quantized_tensor = (quantized_tensor - mean_arr) / std_arr
		quantized_tensor = torch.as_tensor(np.transpose(quantized_tensor, (0, 3, 1, 2)),
											device=DEVICE)
		quantized_outputs = decoder(quantized_tensor)
		quantized_secret_mse += F.mse_loss(quantized_outputs, secrets).item()
		quantized_secret_accuracy += bit_level_accuracy(quantized_outputs, ground_truth)

		for j in range(BATCH_SIZE):

			cv2.imwrite(os.path.join(OUTPUT_PATH, f'batch{i}_{j}.png'),
						cv2.cvtColor(quantized_embeds[j], cv2.COLOR_RGB2BGR))

cover_mse /= i
secret_mse /= i
secret_accuracy /=i
quantized_secret_mse /= i
quantized_secret_accuracy /= i

print(f'Cover MSE: {cover_mse}')
print(f'Secret MSE: {secret_mse}')
print(f'Secret Accuracy: {secret_accuracy}')
print(f'Embed Quantized, Secret MSE: {quantized_secret_mse}')
print(f'Embed Quantized, Secret Accuracy: {quantized_secret_accuracy}')
