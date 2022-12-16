"""
TODO
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
from pathlib import Path
from tqdm import tqdm

from model import Stegnet
from loss import correlation

from my_datasets import dataset_from_text

# assert len(sys.argv) > 1, 'Specify data path.'
# assert len(sys.argv) > 2, 'Specify output path.'
# assert len(sys.argv) > 3, 'Specify loss.'
# assert sys.argv[3] in ('none', 'corr')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Training():
	def __init__(self, batch_size, epochs, six_bit_res):
		args = self.load_args()
		# loading arguments
		self.data_path = args.dataset
		output_path = Path(args.output_dir)
		loss_type = args.loss
		self.load_weights = args.load_weights
		# current timestamp
		self.timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) if not self.load_weights else self.load_weights
		# determine if use correlation loss
		self.corr_loss = correlation if loss_type == 'corr' else None
		# hyperparameters
		self.batch_size = batch_size
		self.epochs = epochs
		self.six_bit_res = six_bit_res
		# create output directory
		if self.load_weights:
			self.output_path = output_path/self.load_weights
			if not self.output_path.is_dir(): raise ValueError('Checkpoint not found.')
		else:
			self.output_path =output_path/self.timestamp
			if not self.output_path.is_dir(): self.output_path.mkdir(parents=True)
		#  tensorboard logger
		self.writer = SummaryWriter(log_dir = self.output_path/'callbacks')

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

		train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
								shuffle=True, drop_last=True)
		val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size,
							shuffle=False, drop_last=True)

		text_data_transform = T.Compose([
			T.ToTensor(),
			# possible values are 0, 32, 160, 192 (divided by 255 I think)
			# so midpoint that we want to be 0 is 96. 96/255 is about 0.376.
			T.Normalize(mean=[0.376, 0.376, 0.376], std=[0.376, 0.376, 0.376])])

		train_text_dataset = dataset_from_text(os.path.join(self.data_path, 'shakespeare_train.txt'),
												(64, 64), self.six_bit_res,
												transform=text_data_transform)
		val_text_dataset = dataset_from_text(os.path.join(self.data_path, 'shakespeare_val.txt'),
												(64, 64), self.six_bit_res,
												transform=text_data_transform)

		train_text_dataloader = DataLoader(train_text_dataset, batch_size=batch_size,
												shuffle=True, drop_last=True)
		val_text_dataloader = DataLoader(val_text_dataset, batch_size=batch_size,
										shuffle=False, drop_last=True)

		return train_dataloader, val_dataloader, train_text_dataloader, val_text_dataloader
	
	def load_ckpts(self, encoder, decoder, optimizer):
		print(f'Loading weights from {self.output_path}')
		all_weights = []
		for file in os.listdir(self.output_path/'weights'):
			if file.endswith('.pt'):
				all_weights.append(file)
		latest_weights = sorted(all_weights)[-1]
		checkpoint = torch.load(self.output_path/'weights'/latest_weights, map_location='cpu')
		encoder.load_state_dict(checkpoint['encoder_dict'])
		decoder.load_state_dict(checkpoint['decoder_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_dict'])
		return encoder, decoder, optimizer, len(all_weights)
	
	def train(self, train_dataloader, val_dataloader,
				train_text_dataloader, val_text_dataloader):
		encoder = Stegnet(6).to(DEVICE)
		decoder = Stegnet(3).to(DEVICE)
		optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()))
		start_epoch = 0
		if self.load_weights:
			encoder, decoder, optimizer, start_epoch = self.load_ckpts(encoder, decoder, optimizer)

		# this ensures each epoch has roughly the same number of examples
		STOP_ITER = len(train_dataloader.dataset) // 2

		for epoch in range(start_epoch, self.epochs):
			# ----------------- training -----------------
			encoder.train()
			decoder.train()

			train_loss = 0.0
			for i, (data, secrets) in tqdm(enumerate(zip(train_dataloader,
															train_text_dataloader),
														start=1)):

				covers, _ = data

				covers = covers.to(DEVICE)
				secrets = secrets.to(DEVICE)

				optimizer.zero_grad()

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
				if i >= STOP_ITER:

					break
			train_loss /= i

			# save weights
			weight_path = self.output_path/"weights"
			weight_path.mkdir(exist_ok=True)
			torch.save({
				'epoch': epoch,
				'loss': train_loss,
				'encoder_dict': encoder.state_dict(),
				'decoder_dict': decoder.state_dict(),
				'optimizer_dict': optimizer.state_dict()
			}, weight_path/f"epoch-{epoch}.pt")

			self.writer.add_scalar('Loss/train', train_loss, epoch)
			# ----------------- validation -----------------
			encoder.eval()
			decoder.eval()
			with torch.no_grad():

				val_loss = 0.0
				for i, (data, secrets) in enumerate(zip(val_dataloader, val_text_dataloader),
													start=1):

					covers, _ = data

					covers = covers.to(DEVICE)
					secrets = secrets.to(DEVICE)

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
	epochs = 100
	six_bit_res = (2, 2)
	train = Training(batch_size, epochs, six_bit_res)

	train_dataloader, val_dataloader,train_text_dataloader, val_text_dataloader = train.data_loader()

	train.train(train_dataloader, val_dataloader, train_text_dataloader, val_text_dataloader)
