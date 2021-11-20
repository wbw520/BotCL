import os
import torch
from termcolor import colored
from configs import parser
from model.model_main import MainModel
from PIL import Image
import torch.nn.functional as F
from torchcam.cams import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM
from loaders.get_loader import load_all_imgs, get_transform
import shutil
from tools import crop_center, make_grad, shot_game
import numpy as np
import copy


shutil.rmtree('vis_compare/', ignore_errors=True)
os.makedirs('vis_compare/', exist_ok=True)


def main():
    model = MainModel(args)
    device = torch.device(args.device)

    # CUDNN
    torch.backends.cudnn.benchmark = True

    checkpoint = torch.load(os.path.join(args.output_dir,
            f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt_no_slot.pt"), map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    print("load pre-trained model finished")

    imgs_database, labels_database, imgs_val, labels_val, cat = load_all_imgs(args)
    transform = get_transform(args)["val"]

    RESNET_CONFIG = dict(input_layer='conv1', conv_layer='back_bone', fc_layer='fc')
    MODEL_CONFIG = {**RESNET_CONFIG}
    conv_layer = MODEL_CONFIG['conv_layer']
    input_layer = MODEL_CONFIG['input_layer']
    fc_layer = MODEL_CONFIG['fc_layer']
    cam_extractors = {"CAM": CAM(model, conv_layer, fc_layer), "GradCAM": GradCAM(model, conv_layer)}
    vis_mode = "CAM"
    record = []
    record_Dauc = []
    for i in range(len(imgs_val)):
        data = imgs_val[i]
        label = labels_val[i]

        image_orl = Image.open(data).convert('RGB').resize([256, 256], resample=Image.BILINEAR)
        if image_orl.mode == 'L':
            image_orl = image_orl.convert('RGB')
        image_orl = crop_center(image_orl, 224, 224)
        imggg = transform(image_orl).unsqueeze(0).to(device)
        output1 = model(imggg, None, None)
        output = F.softmax(output1, dim=-1)
        pred_label = torch.argmax(output).item()
        if pred_label != label:
            print("predict error")
            continue
        print("------------")
        print("The Model Prediction is: ", pred_label)
        print("True is", label)
        mask = make_grad(args, cam_extractors[vis_mode], output1, image_orl, grad_min_level, vis_mode, pred_label)
        hitted, segment = shot_game(mask, data)
        if hitted is None:
            continue
        record.append(hitted)

        record_p = [output[0][pred_label].item()]
        mask1 = mask.flatten()
        ids = np.argsort(-mask1, axis=0)
        for j in range(1, 101, 1):
            thresh = mask1[ids[j * 501]]
            mask_use = copy.deepcopy(mask)
            mask_use[mask_use >= thresh] = 1
            mask_use[mask_use != 1] = 0

            mask_use = torch.from_numpy(mask_use).to(device)
            new_img = imggg * mask_use
            output_c = model(new_img, None, None)
            output_c = F.softmax(output_c, dim=-1)
            record_p.append(output_c[0][pred_label].item())

        record_p = np.array(record_p)
        record_p = (record_p - np.min(record_p)) / (np.max(record_p) - np.min(record_p))
        print(record_p.mean())
        record_Dauc.append(record_p.mean())

    print(np.mean(np.array(record)))
    print(np.mean(np.array(record_Dauc)))
    print(record)
    print(record_Dauc)


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = True
    grad_min_level = 0
    main()
