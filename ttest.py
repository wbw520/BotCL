from torchvision import datasets, transforms
from model.model_main import MainModel
from configs import parser
import torch
import os
from PIL import Image
import numpy as np
from utils import apply_colormap_on_image
from loaders.get_loader import load_all_imgs, get_transform
from tools import for_retrival, attention_estimation
import h5py
from draw_tools import draw_bar, draw_plot
import shutil
import torch.nn.functional as F
from tools import crop_center, shot_game
import cv2
import copy

shutil.rmtree('vis/', ignore_errors=True)
shutil.rmtree('vis_pp/', ignore_errors=True)
os.makedirs('vis/', exist_ok=True)
os.makedirs('vis_pp/', exist_ok=True)
np.set_printoptions(suppress=True)


def main():
    # load all imgs
    imgs_database, labels_database, imgs_val, labels_val, cat = load_all_imgs(args)
    print("All category:")
    print(cat)
    transform = get_transform(args)["val"]

    # load model and weights
    model = MainModel(args, vis=False)
    device = torch.device("cuda:0")
    model.to(device)
    name = f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" + f"{'use_slot_' + args.act_type + '_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"
    print(name)
    checkpoint = torch.load(os.path.join(args.output_dir, name), map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    count = 0

    for i in range(len(imgs_val)):
        print(i)
        data = imgs_val[i]
        label = labels_val[i]
        print(data)
        image_orl = Image.open(data).convert('RGB').resize([256, 256], resample=Image.BILINEAR)
        if image_orl.mode == 'L':
            image_orl = image_orl.convert('RGB')
        image_orl = crop_center(image_orl, 224, 224)
        imggg = transform(image_orl).unsqueeze(0).to(device)
        cpt, pred, att, update = model(imggg)

        pred_label = torch.argmax(pred).item()
        if pred_label == label:
            count += 1
        else:
            print("error")

    print(count/len(imgs_val))


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = False
    main()