# This code is modified from https://github.com/yunhuiguo/CVPR-2021-L2ID-Classification-Challenges/blob/master/datasets/ISIC_few_shot.py

import os

import torchvision
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from PIL import Image


class ISICDataset(Dataset):
    def __init__(self, root_path, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.csv_path = os.path.join(root_path,
            "ISIC2018_Task3_Training_GroundTruth", "ISIC2018_Task3_Training_GroundTruth.csv")
        self.img_path = os.path.join(root_path, "ISIC2018_Task3_Training_Input")

        # Transforms
        if transform == None:
            self.transform = torchvision.transforms.ToTensor()
        else:
            self.transform = transform
        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name = np.asarray(self.data_info.iloc[:, 0])

        self.labels = np.asarray(self.data_info.iloc[:, 1:])
        self.labels = (self.labels!=0).argmax(axis=1)

        self.samples = []
        for name, label in zip(self.image_name, self.labels):
            self.samples.append((name, label))

        # Calculate len
        self.data_len = len(self.image_name)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]
        # Open image
        temp = Image.open(os.path.join(self.img_path, single_image_name + ".jpg"))

        img_as_img = temp.copy()
        # Transform image to tensor
        data = self.transform(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]

        return (data, single_image_label)

    def __len__(self):
        return self.data_len
