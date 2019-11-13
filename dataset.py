import numpy as np
import torch
from torch.utils.data import Dataset
from FreyFaceHelper import FreyFaceHelper

class FreyFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = FreyFaceHelper(root_dir).data
        self.data_size, h, w = self.data.shape
        self.sample_size = h*w
        self.data = self.data.reshape(self.data_size, self.sample_size)
        self.data = torch.from_numpy(self.data).float()
        self.transform = transform

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample