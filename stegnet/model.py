import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
	"""
	Source: https://stackoverflow.com/questions/65154182/implement-separableconv2d-in-pytorch
	"""

	def __init__(self, in_channels, out_channels, kernel_size, padding):

		super().__init__()

		# I'm fairly confident no bias is needed here because these are two linear operators
		self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
									padding=padding, groups=in_channels, bias=False)
		self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

	def forward(self, x):

		x = self.depthwise(x)
		x = self.pointwise(x)
		return x

class Spatial2Channel(nn.Module):
	"""
	Derived based on:
	https://github.com/adamcavendish/Deep-Image-Steganography/blob/master/model/steg_net/steganography.py#L78
	"""

	def __init__(self, activation, in_channels, out_channels, kernel_size):

		super().__init__()

		self.activation = activation
		assert in_channels <= out_channels
		# add padding for extra channels produced and no padding for height and width
		# basically the first in_channels number of channels
		# in the output will have skip connections
		self.padding = (0, out_channels - in_channels, 0, 0, 0, 0)
		self.sep_conv = SeparableConv2d(in_channels, out_channels, kernel_size, padding='same')

	def forward(self, x):

		padded_x = F.pad(x, self.padding, 'constant', 0)

		x = self.activation(x)
		x = self.sep_conv(x)
		# skip connection
		x = x + padded_x

		return x

class Stegnet(nn.Module):

	def __init__(self, in_channels):

		super().__init__()

		self.layer0 = SeparableConv2d(in_channels, 32, 3, padding='same')
		self.layer1 = Spatial2Channel(nn.ELU(), 32, 32, 3)
		self.layer2 = Spatial2Channel(nn.ELU(), 32, 64, 3)
		self.layer3 = Spatial2Channel(nn.ELU(), 64, 64, 3)
		self.layer4 = Spatial2Channel(nn.ELU(), 64, 128, 3)
		self.layer5 = Spatial2Channel(nn.ELU(), 128, 128, 3)
		self.activ5 = nn.ELU()

		self.layer6 = nn.Conv2d(128, 32, 3)
		self.activ6 = nn.ELU()
		self.layer7 = nn.Conv2d(32, 3, 3)

	def forward(self, x):

		x = self.layer0(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.activ5(x)

		x = self.layer6(x)
		x = self.activ6(x)
		x = self.layer7(x)

		return x
