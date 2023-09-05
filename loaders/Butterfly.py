import torch
from PIL import Image
import numpy as np
import os

class Butterfly(torch.utils.data.Dataset):
    def __init__(self, args, phase, transform=None):
        root = args.dataset_dir
        self.paths = []
        self.lbls = []
        cls_names = set()
        for root_dir, _, files in os.walk(os.path.join(root, phase)):
            for fname in files:
                cls_name = root_dir.split(os.path.sep)[-1]
                cls_names.add(cls_name)
                self.lbls.append(cls_name)
                self.paths.append(os.path.join(root_dir, fname))
        
        lbl_map = dict(zip(sorted(list(cls_names)), range(len(list(cls_names)))))
        self.lbls = list(map(lambda x: lbl_map[x], self.lbls))

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.lbls[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(np.array(label))
        return image, label