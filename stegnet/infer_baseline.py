"""
TODO
"""

import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision import datasets

import cv2
from tqdm import tqdm

from model import Stegnet

# to ensure all test images are used, we make sure batch_size * 2 divides dataset length evenly
BATCH_SIZE = 50

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

test_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, 'test'), transform=data_transform)
assert BATCH_SIZE % len(test_dataset) == 0
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2,
								shuffle=False, drop_last=False)

encoder = Stegnet(6).to(DEVICE)
decoder = Stegnet(3).to(DEVICE)

checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')
encoder.load_state_dict(checkpoint['encoder_dict'])
decoder.load_state_dict(checkpoint['decoder_dict'])

cover_mse = 0.0
secret_mse = 0.0
quantized_secret_mse = 0.0
with torch.no_grad():

	for i, data in enumerate(test_dataloader, start=1):

		inputs, _ = data
		inputs = inputs.to(DEVICE)

		covers = inputs[:BATCH_SIZE]
		secrets = inputs[BATCH_SIZE:]
		embeds = encoder(torch.cat([covers, secrets], dim=1))
		outputs = decoder(embeds)

		cover_mse += F.mse_loss(embeds, covers).item()
		secret_mse += F.mse_loss(outputs, secrets).item()

		np_embeds = np.transpose(embeds.detach().cpu().numpy(), (0, 2, 3, 1))
		np_embeds = (np_embeds * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
		np_embeds = np_embeds * 255

		# this is really quantized
		quantized_embeds = np.clip(np_embeds, 0, 255).astype(np.uint8)

		# these are floats, only using quantized as variable name
		quantized_tensor = data_transform(quantized_embeds)
		quantized_outputs = decoder(quantized_tensor)
		quantized_secret_mse += F.mse_loss(quantized_outputs, secrets).item()

		cv2.imwrite(os.path.join(OUTPUT_PATH, f'batch{i}.png'),
					cv2.cvtColor(quantized_embeds, cv2.COLOR_RGB2BGR))

	cover_mse /= i
	secret_mse /= i
	quantized_secret_mse /= i

print(f'Cover MSE: {cover_mse}')
print(f'Secret MSE: {secret_mse}')
print(f'Embed Quantized, Secret MSE: {quantized_secret_mse}')
