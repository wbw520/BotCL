import torch
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split


class Matplot(torch.utils.data.Dataset):
    def __init__(self, data, phase, transform=None):
        self.train, self.val = data
        self.all_data = {"train": self.train, "val": self.val}[phase]
        self.transform = transform

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item_id):
        image_root = self.all_data[item_id][0]
        image = Image.open(image_root).convert('RGB')
        if image.mode == 'L':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        label = self.all_data[item_id][1]
        label = torch.from_numpy(np.array(label))
        return image, label


class MakeImage():
    """
    this class used to make list of data for ImageNet
    """
    def __init__(self):
        self.data_root = "loaders/"

    def get_img(self):
        record = []
        load_npy_y = np.load(self.data_root + "y_data.npy")
        load_npy_cpt = np.load(self.data_root + "concept_data.npy")

        for i in range(load_npy_y.shape[0]):
            img_root = self.data_root + "matplob/raw/" + str(i) + ".jpg"
            record.append([img_root, load_npy_y[i], load_npy_cpt[i]])

        train, test = train_test_split(record, train_size=0.9, random_state=1)
        return train, test
