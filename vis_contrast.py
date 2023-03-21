from torchvision import datasets, transforms
from model.contrast.model_main import MainModel
from configs import parser
import torch
import os
from PIL import Image
import numpy as np
from utils.tools import apply_colormap_on_image
from loaders.get_loader import load_all_imgs, get_transform
from utils.tools import for_retrival, attention_estimation
import h5py
from utils.draw_tools import draw_bar, draw_plot
import shutil
from utils.tools import crop_center
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
    transform = get_transform(args)["val"]

    # load model and weights
    model = MainModel(args, vis=True)
    device = torch.device("cuda:0")
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir,
            f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" +
    f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"), map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    index = args.index

    # # attention statistic
    # name = "Yellow_headed_Blackbird"
    # att_record = attention_estimation(imgs_database, labels_database, model, transform, device, name=name)
    # draw_plot(att_record, name)

    data = imgs_database[index]
    label = labels_database[index]

    print("-------------------------")
    print("label true is: ", cat[label])
    print("-------------------------")
    # data = "/home/wangbowen/DATA/ImageNet/ILSVRC/Data/CLS-LOC/train/n01494475/n01494475_618.JPEG"
    if args.dataset == "MNIST":
        img_orl = Image.fromarray(data.numpy()).resize([224, 224], resample=Image.BILINEAR)
    elif args.dataset == "cifar10":
        img_orl = Image.fromarray(data).resize([224, 224], resample=Image.BILINEAR)
    else:
        img_orl = Image.open(data).convert('RGB').resize([256, 256], resample=Image.BILINEAR)

    img_orl2 = crop_center(img_orl, 224, 224)
    img_orl2.save(f'vis/origin.png')
    cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, None)
    print("-------------------------")
    pp = torch.argmax(pred, dim=-1)
    print("predicted as: ", cat[pp])

    w = model.state_dict()["cls.weight"][label]
    w_numpy = np.around(torch.tanh(w).cpu().detach().numpy(), 4)
    ccc = np.around(cpt.cpu().detach().numpy(), 4)
    # draw_bar(w_numpy, name)

    print("--------weight---------")
    print(w_numpy)

    print("--------cpt---------")
    print(ccc)

    print("------sum--------")
    print((ccc/2 + 0.5) * w_numpy)
    # if args.use_weight:
    #     w[w < 0] = 0
    #     cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), w)

    for id in range(args.num_cpt):
        print("-------------")
        # slot_image = np.array(Image.open(f'vis/0_slot_{id}.png'))
        slot_image = cv2.imread(f'vis/0_slot_{id}.png', cv2.IMREAD_GRAYSCALE)
        heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl2, slot_image, 'jet')
        heatmap_on_image.save("vis/" + f'0_slot_mask_{id}.png')

    # get retrieval cases
    f1 = h5py.File(f"data_map/{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_{args.cpt_activation}.hdf5", 'r')
    database_hash = f1["database_hash"]
    database_labels = f1["database_labels"]
    test_hash = f1["test_hash"]
    test_labels = f1["test_labels"]

    # query_sample = np.array([database_hash[index]])
    # query_sample[0][location] = -1
    # ids = for_retrival(args, database_hash, query_sample, None)
    #
    # for i in range(len(ids)):
    #     current_is = ids[i]
    #     img_re = Image.open(imgs_database[current_is]).convert('RGB').resize([224, 224], resample=Image.BILINEAR)
    #     img_re.save(f"retrieval_results/re_{i}.png")

    print("-------------------------")
    print("generating concept samples")

    for j in range(args.num_cpt):
        root = 'vis_pp/' + "cpt" + str(j) + "/"
        os.makedirs(root, exist_ok=True)
        selected = np.array(database_hash)[:, j]
        ids = np.argsort(-selected, axis=0)
        idx = ids[:args.top_samples]
        for i in range(len(idx)):
            current_is = idx[i]
            category = cat[int(database_labels[current_is][0])]
            if args.dataset == "MNIST":
                img_orl = Image.fromarray(imgs_database[current_is].numpy())
            elif args.dataset == "cifar10":
                img_orl = Image.fromarray(imgs_database[current_is])
            else:
                img_orl = Image.open(imgs_database[current_is]).convert('RGB')
            img_orl = img_orl.resize([256, 256], resample=Image.BILINEAR)
            img_orl2 = crop_center(img_orl, 224, 224)
            cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, [i, category, j])
            img_orl2.save(root + f'/orl_{i}_{category}.png')
            slot_image = np.array(Image.open(root + f'mask_{i}_{category}.png'), dtype=np.uint8)
            heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl2, slot_image, 'jet')
            heatmap_on_image.save(root + f'jet_{i}_{category}.png')


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = False
    main()