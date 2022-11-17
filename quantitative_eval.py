from loaders.matplob import MakeImage
import torch
from configs import parser
import os
import numpy as np
from loaders.get_loader import get_transformations_synthetic
from PIL import Image
from model.retrieval.model_main import MainModel
import cv2
from loaders.ImageNet import get_name
import json


def make_statistic(cpt_nums):
    record = []
    for i in range(cpt_nums):
        record.append({0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
    return record


def main():
    model = MainModel(args, vis=True)
    device = torch.device(args.device)
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir,
            f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" +
    f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}_version3.pt"), map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    transform = get_transformations_synthetic()
    data_ = MakeImage().get_img()[1]
    statistic = make_statistic(cpt_num)
    statistic_sample = [0, 0, 0, 0, 0]

    for i in range(len(data_)):
        if i % 10 == 0:
            print("processed " + str(i) + " samples")
        root = data_[i][0]
        img_orl = Image.open(root).convert('RGB')
        name_label = root.split("/")[-1].split(".")[0]
        img_orl = img_orl.resize([224, 224], resample=Image.BILINEAR)
        cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, None)
        sample_root = "loaders/matplob/label/" + name_label
        cpt_sample = get_name(sample_root, mode_folder=False)

        if cpt_sample is None:
            print("not exist folder " + sample_root)
            continue

        for k in range(len(cpt_sample)):
            sample_indexs = int(cpt_sample[k].split(".")[0])
            statistic_sample[sample_indexs] += 1

        for j in range(cpt_num):
            mask_current = np.array(Image.open("vis/0_slot_" + str(j) + ".png"))
            MAX = np.max(mask_current)
            MIN = np.min(mask_current)
            if MAX < 10:
                continue
            mask_current = (mask_current - MIN) / (MAX - MIN)
            upper = mask_current > thresh_att
            lower = mask_current <= thresh_att
            mask_current[upper] = 1
            mask_current[lower] = 0

            for s in range(len(cpt_sample)):
                sample_index = int(cpt_sample[s].split(".")[0])

                current_sample = cv2.imread(sample_root + "/" + cpt_sample[s], 0)
                current_sample[current_sample != 255] = 1
                current_sample[current_sample == 255] = 0

                overlap = mask_current + current_sample
                overlap_sum = (overlap == 2).sum()
                union_sum = current_sample.sum()

                if overlap_sum / union_sum > thresh_overlap:
                    statistic[j][sample_index] += 1

    # draw_syn(statistic, statistic_sample)
    # for l in range(len(statistic)):
    #     print("cpt ", l)
    #     print(statistic[l])
    #
    # print(statistic_sample)

    with open("cpt_save2.json", "w") as write_file:
        json.dump({"files": statistic}, write_file)


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = False
    args.batch_size = 1
    thresh_att = 0.5
    thresh_overlap = 0.2
    cpt_num = args.num_cpt
    main()

