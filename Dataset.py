from __future__ import print_function
import torch
import pandas as pd
import torch.utils.data as data
import numpy as np
from torchvision.io import read_image
import torchvision.transforms as transforms
from PIL import Image


class OfficeHome(data.Dataset):
    def __init__(self, split, domain, transform=None, target_transform=None):
        self.data_info = pd.read_csv("data_info.csv", index_col=0)
        self.data_info = self.data_info[
            (self.data_info["split"] == split) & (self.data_info["domain"] == domain)
        ]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        img_path = self.data_info.iloc[idx, 0]
        # image = read_image(img_path)
        # image = transforms.ToPILImage()(image)
        with open(img_path, "rb") as f:
            with Image.open(f) as imgf:
                image = imgf.convert("RGB")

        label = self.data_info.iloc[idx, -1]
        # label = torch.LongTensor([np.int64(label).item()])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.data_info)
