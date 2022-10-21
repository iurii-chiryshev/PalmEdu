import numpy as np
import cv2
import sys
from torch.utils import data
from torch.utils.data import DataLoader
import json
import glob
import pathlib
import os
from PIL import Image
from torchvision import transforms
import pandas as pd

class AngleDatasets(data.Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms
        df = pd.read_csv(os.path.join(path,"label.csv"),sep=";")
        self.list = df.values.tolist()

    def __getitem__(self, index):
        row = self.list[index]
        file_name = row[0]
        label = row[1]
        img = cv2.imread(os.path.join(self.path, file_name),cv2.IMREAD_COLOR)

        if self.transforms:
            [img, label] = self.transforms(img,label)
        return img, label

    def __len__(self):
        return len(self.list)


if __name__ == '__main__':
    path = r"D:\projects\PalmEduData\train"
    datset = AngleDatasets(path)
    dataloader = DataLoader(datset, batch_size=1, shuffle=False, num_workers=1)

    for img, label in dataloader:
        print("image", img.dtype)
        print("angle", label.dtype)
    print(len(datset))
