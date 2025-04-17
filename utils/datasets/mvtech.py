from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MVTechDataset_AD(Dataset):
    
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(self.samples[idx], tuple):
            img_path, mask_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            mask = plt.imread(mask_path)
            mask = np.expand_dims(mask, axis=0)
        else:
            image = Image.open(self.samples[idx]).convert('RGB')
            mask = np.zeros((1, image.size[0], image.size[1]))
            label = 0

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, mask, label

class MVTechDataset_cls(Dataset):
    
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            images = []
            labels = []
            for i in range(idx.start, idx.stop, idx.step if idx.step else 1):
                image, label = self.__getitem__(i)
                images.append(image)
                labels.append(label)
            return np.array(images), np.array(labels)
        else:
            img_path = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            np_image = np.array(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return np_image, self.labels[idx]