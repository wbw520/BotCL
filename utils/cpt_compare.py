from model.retrieval.model_main import MainModel
from configs import parser
import torch
import os
import torch.nn.functional as F
from PIL import Image
from loaders.get_loader import load_all_imgs, get_transform, AddGaussianNoise
import shutil
import numpy as np
from torchvision import datasets, transforms
import cv2


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

    # load model and weights
    model = MainModel(args, vis=True)
    device = torch.device("cuda:0")
    model.to(device)
    name = f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" + f"{'use_slot_' + args.act_type + '_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"
    checkpoint = torch.load(os.path.join("../saved_model", name), map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    record = []
    record_Dauc = []

    lip_record = []

    trans1 = get_transform(args)["val"]
    trans2 = transforms.Compose([AddGaussianNoise(0., 1.)])

    for i in range(len(imgs_val)):
        print(i)
        model.vis = True
        data = imgs_val[i]
        label = labels_val[i]
        # print(i)
        # print(data)

        image_orl = Image.open(data).convert('RGB').resize([256, 256], resample=Image.BILINEAR)
        if image_orl.mode == 'L':
            image_orl = image_orl.convert('RGB')

        img1 = trans1(image_orl).unsqueeze(0).to(device)
        w1 = model.state_dict()["cls.weight"][label]
        w11 = w1.clone()
        w11 = torch.relu(w11)
        cpt1, pred1, att1, update1 = model(img1, [w11, "1"])

        pred1 = F.softmax(pred1, dim=-1)
        pred_label1 = torch.argmax(pred1).item()
        if pred_label1 != label:
            print("predict error")
            continue

        mask1 = cv2.imread("vis/" + f'overall_1.png', cv2.IMREAD_UNCHANGED) / 255

        img2 = trans2(trans1(image_orl)).unsqueeze(0).to(device)
        cpt2, pred2, att2, update2 = model(img2, [w11, "2"])

        # pred2 = F.softmax(pred2, dim=-1)
        # pred_label1 = torch.argmax(pred2).item()
        # if pred_label1 != label:
        #     print("predict error")
        #     continue

        mask2 = cv2.imread("vis/" + f'overall_2.png', cv2.IMREAD_UNCHANGED) / 255

        dist_f = np.linalg.norm(x=(mask2-mask1).flatten(), ord=2)
        dist_x = (img2-img1).norm().cpu().detach().numpy()

        lip = dist_f/dist_x
        lip_record.append(lip)


    lip_record = np.array(lip_record)
    print(np.mean(lip_record))
    print(np.std(lip_record))
        # hitted, segment = shot_game(mask1, data)
        # if hitted is None:
        #     continue
        # record.append(hitted)

    #     record_p = [pred[0][pred_label].item()]
    #     mask1 = mask.flatten()
    #     ids = np.argsort(-mask1, axis=0)
    #     model.vis = False
    #     for j in range(1, 101, 1):
    #         thresh = mask1[ids[j * 501]]
    #         mask_use = copy.deepcopy(mask)
    #         mask_use[mask_use >= thresh] = 0
    #         mask_use[mask_use != 0] = 1
    #
    #         mask_use = torch.from_numpy(mask_use).to(device, torch.float32)
    #         new_img = img1 * mask_use
    #         cpt, pred, att, update = model(new_img, None, None)
    #         output_c = F.softmax(pred, dim=-1)
    #         record_p.append(output_c[0][pred_label].item())
    #     record_p = np.array(record_p)
    #     record_p = (record_p - np.min(record_p)) / (np.max(record_p) - np.min(record_p))
    #     # print(record_p)
    #     print(record_p.mean())
    #     record_Dauc.append(record_p.mean())
    #
    # print(np.mean(np.array(record)))
    # print(record)
    # print(np.mean(np.array(record_Dauc)))


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = False
    main()