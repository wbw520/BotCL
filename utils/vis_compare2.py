import os
import torch
from termcolor import colored
from configs import parser
from model.retrieval.model_main import MainModel
from PIL import Image
import torch.nn.functional as F
from torchcam.cams import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM
from loaders.get_loader import load_all_imgs, get_transform, AddGaussianNoise
import shutil
from tools import crop_center, make_grad, shot_game
from torchvision import datasets, transforms
import numpy as np
import copy
from torchray.attribution.rise import rise


shutil.rmtree('vis_compare/', ignore_errors=True)
os.makedirs('vis_compare/', exist_ok=True)


def rise_cal(model, image, target_index):
    mask = rise(model, image, target_index)
    mask = mask.cpu().numpy()[0, 0]

    mask = np.maximum(mask, 0)
    mask = mask - np.min(mask)
    mask = mask / np.max(mask)
    mask = np.maximum(mask, grad_min_level)
    mask = mask - np.min(mask)
    mask = mask / np.max(mask)
    return mask


def main():
    model = MainModel(args)
    device = torch.device(args.device)

    # CUDNN
    torch.backends.cudnn.benchmark = True

    trans1 = get_transform(args)["val"]
    trans2 = transforms.Compose([AddGaussianNoise(0., 1.)])

    checkpoint = torch.load(os.path.join("../saved_model", f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt_no_slot.pt"), map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    print("load pre-trained model finished")

    imgs_database, labels_database, imgs_val, labels_val, cat = load_all_imgs(args)
    transform = get_transform(args)["val"]

    record = []
    record_Dauc = []
    lip_record = []

    for i in range(len(imgs_val)):
        print(i)
        data = imgs_val[i]
        label = labels_val[i]

        image_orl = Image.open(data).convert('RGB').resize([256, 256], resample=Image.BILINEAR)
        if image_orl.mode == 'L':
            image_orl = image_orl.convert('RGB')

        img1 = trans1(image_orl).unsqueeze(0).to(device)
        output1 = model(img1, None, None)
        output = F.softmax(output1, dim=-1)
        pred_label = torch.argmax(output).item()
        if pred_label != label:
            print("predict error")
            continue
        # print("------------")
        # print("The Model Prediction is: ", pred_label)
        # print("True is", label)

        mask1 = rise_cal(model, img1, pred_label)

        img2 = trans2(trans1(image_orl)).unsqueeze(0).to(device)

        mask2 = rise_cal(model, img2, pred_label)

        dist_f = np.linalg.norm(x=(mask2 - mask1).flatten(), ord=2)
        dist_x = (img2 - img1).norm().cpu().detach().numpy()

        lip = dist_f / dist_x
        lip_record.append(lip)

    lip_record = np.array(lip_record)
    print(np.mean(lip_record))
    print(np.std(lip_record))


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = True
    grad_min_level = 0
    main()
