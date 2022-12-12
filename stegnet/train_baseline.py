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
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from model import Stegnet
from loss import correlation

assert len(sys.argv) > 1, 'Specify data path.'
assert len(sys.argv) > 2, 'Specify output path.'
assert len(sys.argv) > 3, 'Specify loss.'
assert sys.argv[3] in ('none', 'corr')

# tensorboard logger
writer = SummaryWriter()

BATCH_SIZE = 64
EPOCHS = 10

CORR_LOSS = correlation if sys.argv[3] == 'corr' else None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

the_transform = T.Compose([
	T.ToTensor(),
	T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(sys.argv[1], 'train'), transform=the_transform)
val_dataset = datasets.ImageFolder(os.path.join(sys.argv[1], 'val'), transform=the_transform)
# BATCH_SIZE * 2 because we divide data into pairs
# it's very important to drop last to always ensure we can divide by 2
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE * 2,
								shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
							shuffle=False, drop_last=True)

encoder = Stegnet(6).to(DEVICE)
decoder = Stegnet(3).to(DEVICE)

optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()))

if len(sys.argv) > 4:

	checkpoint = torch.load(sys.argv[3], map_location='cpu')
	encoder.load_state_dict(checkpoint['encoder_dict'])
	decoder.load_state_dict(checkpoint['decoder_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_dict'])

for epoch in range(EPOCHS):

	encoder.train()
	decoder.train()

	train_loss = 0.0
	for i, data in tqdm(enumerate(train_dataloader, start=1)):

		inputs, _ = data
		inputs = inputs.to(DEVICE)

		optimizer.zero_grad()

		covers = inputs[:BATCH_SIZE]
		secrets = inputs[BATCH_SIZE:]
		embeds = encoder(torch.cat([covers, secrets], dim=1))
		outputs = decoder(embeds)

		loss = (F.l1_loss(embeds, covers) + F.l1_loss(outputs, secrets) +
				torch.mean(torch.var(embeds - covers, dim=(1, 2, 3), unbiased=True)) +
				torch.mean(torch.var(outputs - secrets, dim=(1, 2, 3), unbiased=True)))
		if CORR_LOSS is not None:

			loss += CORR_LOSS(embeds - covers, secrets)

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
	}, os.path.join(sys.argv[2], f"epoch-{epoch + 1}.pt"))

	writer.add_scalar('Loss/train', train_loss, epoch)

	encoder.eval()
	decoder.eval()

	with torch.no_grad():

		val_loss = 0.0
		for i, data in enumerate(val_dataloader, start=1):

			inputs, _ = data
			inputs = inputs.to(DEVICE)

			covers = inputs[:BATCH_SIZE]
			secrets = inputs[BATCH_SIZE:]
			embeds = encoder(torch.cat([covers, secrets], dim=1))
			outputs = decoder(embeds)

			loss = (F.l1_loss(embeds, covers) + F.l1_loss(outputs, secrets) +
					torch.mean(torch.var(embeds - covers, dim=(1, 2, 3), unbiased=True)) +
					torch.mean(torch.var(outputs - secrets, dim=(1, 2, 3), unbiased=True)))
			if CORR_LOSS is not None:

				loss += CORR_LOSS(embeds - covers, secrets)

			val_loss += loss.item()
		val_loss /= i
	writer.add_scalar('Loss/val', val_loss, epoch)

	print(f"epoch={epoch};train_loss={train_loss};val_loss={val_loss}")
	writer.close()
