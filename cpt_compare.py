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
from captum.metrics import infidelity


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
    model = MainModel(args, vis=True)
    device = torch.device("cuda:0")
    model.to(device)
    name = f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" + f"{'use_slot_' + args.act_type + '_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"
    print(name)
    checkpoint = torch.load(os.path.join(args.output_dir, name), map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    record = []
    record_Dauc = []

    for i in range(len(imgs_val)):
        print(i)
        model.vis = True
        data = imgs_val[i]
        print(data)
        label = labels_val[i]
        # print(i)
        # print(data)

        image_orl = Image.open(data).convert('RGB').resize([256, 256], resample=Image.BILINEAR)
        if image_orl.mode == 'L':
            image_orl = image_orl.convert('RGB')
        image_orl = crop_center(image_orl, 224, 224)
        imggg = transform(image_orl).unsqueeze(0).to(device)
        w = model.state_dict()["cls.weight"][label]
        w2 = w.clone()
        w2 = torch.relu(w2)
        cpt, pred, att, update = model(imggg, w2)

        pred = F.softmax(pred, dim=-1)
        pred_label = torch.argmax(pred).item()
        if pred_label != label:
            print("predict error")
            continue

        # print("------------")
        # print("The Model Prediction is: ", pred_label)
        # print("True is", label)

        # for id in range(args.num_cpt):
        #     slot_image = np.array(Image.open(f'vis/0_slot_{id}.png'), dtype=np.uint8)
        #     heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl2, slot_image, 'jet')
        #     heatmap_on_image.save("vis/" + f'0_slot_mask_{id}.png')

        # slot_image = np.array(Image.open(f'vis/overall.png'), dtype=np.uint8)
        # heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl2, slot_image, 'jet')
        # heatmap_on_image.save("vis/" + f'overall_mask.png')
        mask = cv2.imread("vis/" + f'overall.png', cv2.IMREAD_UNCHANGED) / 255
        hitted, segment = shot_game(mask, data)
        if hitted is None:
            continue
        record.append(hitted)

        record_p = [pred[0][pred_label].item()]
        mask1 = mask.flatten()
        ids = np.argsort(-mask1, axis=0)
        model.vis = False
        for j in range(1, 101, 1):
            thresh = mask1[ids[j * 501]]
            mask_use = copy.deepcopy(mask)
            mask_use[mask_use >= thresh] = 0
            mask_use[mask_use != 0] = 1

            mask_use = torch.from_numpy(mask_use).to(device, torch.float32)
            new_img = imggg * mask_use
            cpt, pred, att, update = model(new_img, None, None)
            output_c = F.softmax(pred, dim=-1)
            record_p.append(output_c[0][pred_label].item())
        record_p = np.array(record_p)
        record_p = (record_p - np.min(record_p)) / (np.max(record_p) - np.min(record_p))
        # print(record_p)
        print(record_p.mean())
        record_Dauc.append(record_p.mean())

    print(np.mean(np.array(record)))
    print(record)
    print(np.mean(np.array(record_Dauc)))


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = False
    main()