import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, csv_path, train=True):
        self.df = pd.read_csv(csv_path)
        self.train = train

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        img = cv2.imread(row["filepath"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0

        # Simple augmentation
        if self.train:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()

        # Normalize (ImageNet)
        img = (img - self.mean) / self.std

        img = np.transpose(img, (2, 0, 1))

        label = int(row["label"])

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label)

