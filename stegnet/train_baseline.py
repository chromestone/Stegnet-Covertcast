"""
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

assert len(sys.argv) > 1, 'Specify data path.'
assert len(sys.argv) > 2, 'Specify output path.'

# tensorboard logger
writer = SummaryWriter()

BATCH_SIZE = 64
EPOCHS = 10

the_transform = T.Compose([
	T.ToTensor(),
	T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(sys.argv[1], 'train'), transform=the_transform)
val_dataset = datasets.ImageFolder(os.path.join(sys.argv[1], 'val'), transform=the_transform)
# BATCH_SIZE * 2 because we divide data into pairs
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE * 2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

encoder = Stegnet(6)
decoder = Stegnet(3)

optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()))

if len(sys.argv) > 3:

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

		optimizer.zero_grad()

		covers = inputs[:BATCH_SIZE]
		secrets = inputs[BATCH_SIZE:]
		embeds = encoder(torch.cat([covers, secrets], dim=1))
		outputs = decoder(embeds)

		loss = (F.l1_loss(embeds, covers) + F.l1_loss(outputs, secrets) +
				torch.mean(torch.var(embeds - covers, dim=(1, 2, 3), unbiased=True)) +
				torch.mean(torch.var(outputs - secrets, dim=(1, 2, 3), unbiased=True)))
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

	writer.add_scalar('Loss/train', train_loss.item(), epoch)

	encoder.eval()
	decoder.eval()

	with torch.no_grad():

		val_loss = 0.0
		for i, data in enumerate(val_dataloader, start=1):

			inputs, _ = data

			covers = inputs[:BATCH_SIZE]
			secrets = inputs[BATCH_SIZE:]
			embeds = encoder(torch.cat([covers, secrets], dim=1))
			outputs = decoder(embeds)

			loss = (F.l1_loss(embeds, covers) + F.l1_loss(outputs, secrets) +
					torch.mean(torch.var(embeds - covers, dim=(1, 2, 3), unbiased=True)) +
					torch.mean(torch.var(outputs - secrets, dim=(1, 2, 3), unbiased=True)))

			val_loss += loss.item()
		val_loss /= i
	writer.add_scalar('Loss/val', val_loss.item(), epoch)

	print(f"epoch={epoch};train_loss={train_loss};val_loss={train_loss}")
	writer.close()
