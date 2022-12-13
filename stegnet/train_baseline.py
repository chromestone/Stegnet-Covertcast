"""

Usage:
mkdir ckpt_dir
python3 train_baseline.py tiny-imagenet-200 ckpt_dir none [restart ckpt]
python3 train_baseline.py tiny-imagenet-200 ckpt_dir corr [restart ckpt]

Sources:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f
https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""
from itertools import chain
import os, argparse
import sys, time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from model import Stegnet
from loss import correlation

# assert len(sys.argv) > 1, 'Specify data path.'
# assert len(sys.argv) > 2, 'Specify output path.'
# assert len(sys.argv) > 3, 'Specify loss.'
# assert sys.argv[3] in ('none', 'corr')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Training():
	def __init__(self, batch_size, epochs):
		args = self.load_args()
		# loading arguments
		self.data_path = args.dataset
		output_path = args.output_dir
		loss_type = args.loss
		self.load_weights = args.load_weights
		# current timestamp
		self.timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) if not self.load_weights else self.load_weights
		# determine if use correlation loss
		self.corr_loss = correlation if loss_type == 'corr' else None
		# hyperparameters
		self.batch_size = batch_size
		self.epochs = epochs
		# create output directory
		if self.load_weights:
			self.output_path = os.path.join(output_path, self.load_weights)
			if not os.path.isdir(self.output_path): raise ValueError('Checkpoint not found.')
		else:
			self.output_path = os.path.join(output_path, self.timestamp)
			if not os.path.exists(self.output_path): os.makedirs(self.output_path)
		#  tensorboard logger
		self.writer = SummaryWriter(log_dir = self.output_path)

	def load_args(self):
		parser = argparse.ArgumentParser()
		parser.add_argument('--dataset', type=str, help='Path to data.', default='tiny-imagenet-200')
		parser.add_argument('--output_dir', type=str, help='Path to output.', default='output_dir')
		parser.add_argument('--loss', type=str, help='Loss to use.')
		parser.add_argument('--load_weights', type=str, help='Restart from checkpoint with timestamp.', required=False)
		args = parser.parse_args()
		return args
	
	def data_loader(self):
		data_transform = T.Compose([
			T.ToTensor(),
			T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

		train_dataset = datasets.ImageFolder(os.path.join(self.data_path, 'train'), transform=data_transform)
		val_dataset = datasets.ImageFolder(os.path.join(self.data_path, 'val'), transform=data_transform)

		# BATCH_SIZE * 2 because we divide data into pairs
		# it's very important to drop last to always ensure we can divide by 2
		train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size * 2,
								shuffle=True, drop_last=True)
		val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size * 2,
							shuffle=False, drop_last=True)
		return train_dataloader, val_dataloader
	
	def load_ckpts(self, encoder, decoder, optimizer):
		weight_path = os.path.join(self.output_path, self.load_weights)
		if not os.path.exists(weight_path):
			raise ValueError('Checkpoint not found.')
		checkpoint = torch.load(weight_path, map_location='cpu')
		encoder.load_state_dict(checkpoint['encoder_dict'])
		decoder.load_state_dict(checkpoint['decoder_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_dict'])
		return encoder, decoder, optimizer
	
	def train(self, train_dataloader, val_dataloader):
		encoder = Stegnet(6).to(DEVICE)
		decoder = Stegnet(3).to(DEVICE)
		optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()))

		if self.load_weights:
			encoder, decoder, optimizer = self.load_ckpts(encoder, decoder, optimizer)

		for epoch in range(self.epochs):
			# ----------------- training -----------------
			encoder.train()
			decoder.train()

			train_loss = 0.0
			for i, data in tqdm(enumerate(train_dataloader, start=1)):

				inputs, _ = data
				inputs = inputs.to(DEVICE)

				optimizer.zero_grad()

				covers = inputs[:self.batch_size]
				secrets = inputs[self.batch_size:]
				embeds = encoder(torch.cat([covers, secrets], dim=1))
				outputs = decoder(embeds)

				loss = (F.l1_loss(embeds, covers) + F.l1_loss(outputs, secrets) +
						torch.mean(torch.var(embeds - covers, dim=(1, 2, 3), unbiased=True)) +
						torch.mean(torch.var(outputs - secrets, dim=(1, 2, 3), unbiased=True)))
				if self.corr_loss is not None:
					loss += torch.abs(self.corr_loss(embeds - covers, secrets))

				loss.backward()
				optimizer.step()
				train_loss += loss.item()
			train_loss /= i

			torch.save({
				'epoch': epoch + 1,
				'loss': train_loss,
				'encoder_dict': encoder.state_dict(),
				'decoder_dict': decoder.state_dict(),
				'optimizer_dict': optimizer.state_dict()
			}, os.path.join(self.output_path, f"epoch-{epoch + 1}.pt"))

			self.writer.add_scalar('Loss/train', train_loss, epoch)
			# ----------------- validation -----------------
			encoder.eval()
			decoder.eval()
			with torch.no_grad():

				val_loss = 0.0
				for i, data in enumerate(val_dataloader, start=1):

					inputs, _ = data
					inputs = inputs.to(DEVICE)

					covers = inputs[:self.batch_size]
					secrets = inputs[self.batch_size:]
					embeds = encoder(torch.cat([covers, secrets], dim=1))
					outputs = decoder(embeds)

					loss = (F.l1_loss(embeds, covers) + F.l1_loss(outputs, secrets) +
							torch.mean(torch.var(embeds - covers, dim=(1, 2, 3), unbiased=True)) +
							torch.mean(torch.var(outputs - secrets, dim=(1, 2, 3), unbiased=True)))
					if self.corr_loss is not None:
						loss += torch.abs(self.corr_loss(embeds - covers, secrets))
					val_loss += loss.item()
				val_loss /= i
			self.writer.add_scalar('Loss/val', val_loss, epoch)

			print(f"epoch={epoch};train_loss={train_loss};val_loss={val_loss}")
			self.writer.close()


if __name__ == '__main__':
	batch_size = 64
	epochs = 1
	train = Training(batch_size, epochs)
	train_dataloader, val_dataloader = train.data_loader()
	train.train(train_dataloader, val_dataloader)

