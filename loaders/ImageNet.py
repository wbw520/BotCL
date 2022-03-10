import torch
from PIL import Image
import numpy as np
import os


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return sorted(dirs)
        else:
            return sorted(file)


class ImageNet(torch.utils.data.Dataset):
    def __init__(self, args, phase, transform=None):
        self.train, self.val, self.category = MakeImage(args).get_data()
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
    def __init__(self, args):
        self.image_root = os.path.join(args.dataset_dir, args.dataset, "ILSVRC/Data/CLS-LOC")
        self.category = get_name(self.image_root + "/train/")
        self.used_cat = self.category[:args.num_classes]

    def get_data(self):
        train = self.get_img(self.used_cat, "train")
        val = self.get_img(self.used_cat, "val")
        return train, val, self.used_cat

    def get_img(self, folders, phase):
        record = []
        for folder in folders:
            current_root = os.path.join(self.image_root, phase, folder)
            images = get_name(current_root, mode_folder=False)
            for img in images:
                record.append([os.path.join(current_root, img), self.deal_label(folder)])
        return record

    def deal_label(self, img_name):
        back = self.used_cat.index(img_name)
        return back
