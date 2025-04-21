# loss.py

import torch

def get_loss(name='cross_entropy'):
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Loss function '{name}' is not implemented.")
    elif name == 'focal':
    	return FocalLoss()