"""
This module contains loss or loss related functions.
"""

import torch

def correlation(X, Y):
	"""
	Computes correlation for batch of image like tensors.
	"""

	cov = torch.mean((X - torch.mean(X, (1, 2, 3), True)) * (Y - torch.mean(Y, (1, 2, 3), True)),
						(1, 2, 3))
	corr = cov / torch.sqrt(torch.var(X, dim=(1, 2, 3), unbiased=True) *
							torch.var(Y, dim=(1, 2, 3), unbiased=True))
	return torch.mean(corr)
