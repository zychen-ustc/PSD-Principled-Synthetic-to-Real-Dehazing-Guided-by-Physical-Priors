import torch.utils.data as data
import os
from PIL import Image
from random import randrange
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize


class ConcatDataset(data.Dataset):

    def __init__(self, dataloader_syn, dataloader_real):

        super().__init__()
        self.datasets = (dataloader_syn, dataloader_real)

    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
    
class ConcatDataset3(data.Dataset):

    def __init__(self, dataloader_syn, dataloader_ITS, dataloader_real):

        super().__init__()
        self.datasets = (dataloader_syn, dataloader_ITS, dataloader_real)

    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)