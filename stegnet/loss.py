"""
TODO
"""

import torch

def correlation(X, Y):

	cov = torch.mean((X - torch.mean(X, (1, 2, 3), True)) * (Y - torch.mean(Y, (1, 2, 3), True)),
						(1, 2, 3))
	corr = cov / (torch.var(X, dim=(1, 2, 3), unbiased=True) *
					torch.var(Y, dim=(1, 2, 3), unbiased=True))
	return torch.mean(corr)
