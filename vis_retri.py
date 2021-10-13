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
    f"{'use_slot_' + args.act_type + '_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"), map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    # # 0, 17   7
    # # 7, 9, 16, 12   9
    # # 15    5
    # # 4    4

    # 153  5
    index = 330

    # attention statistic
    att_record = attention_estimation(imgs_database, labels_database, model, transform, device)
    print(att_record.shape)
    draw_plot(att_record)

    # data = imgs_val[index]
    # label = labels_val[index]
    data = imgs_database[index]
    label = labels_database[index]
    print("-------------------------")
    print("label true is: ", cat[label])
    print("-------------------------")
    if args.dataset == "MNIST":
        img_orl = Image.fromarray(data.numpy()).resize([224, 224], resample=Image.BILINEAR)
    elif args.dataset == "cifar10":
        img_orl = Image.fromarray(data).resize([224, 224], resample=Image.BILINEAR)
    else:
        img_orl = Image.open(data).convert('RGB').resize([224, 224], resample=Image.BILINEAR)
    img_orl.save(f'vis/origin.png')
    cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, None)
    print("-------------------------")
    pp = torch.argmax(pred, dim=-1)
    print("predicted as: ", cat[pp])
    print("--------weight---------")
    w = model.state_dict()["cls.weight"][label]
    w_numpy = np.around(torch.tanh(w).cpu().detach().numpy(), 4)
    draw_bar(w_numpy)
    print(w_numpy)
    print("--------cpt---------")
    if args.cpt_activation == "att":
        cpt_act = att
    else:
        cpt_act = update
    print(np.around(torch.sum(cpt_act, dim=-1).cpu().detach().numpy(), 4))
    if args.weight_att:
        w[w < 0] = 0
        cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), w)

    for id in range(args.num_cpt):
        slot_image = np.array(Image.open(f'vis/0_slot_{id}.png'), dtype=np.uint8)
        heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl, slot_image, 'jet')
        heatmap_on_image.save("vis/" + f'0_slot_mask_{id}.png')

    # get retrieval cases
    f1 = h5py.File(f"data_map/{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_{args.act_type}_{args.cpt_activation}.hdf5", 'r')
    database_hash = f1["database_hash"]
    database_labels = f1["database_labels"]
    test_hash = f1["test_hash"]
    test_labels = f1["test_labels"]

    query = np.zeros((1, args.num_cpt)) - 1
    location = 4
    query[0][location] = 1
    ids = for_retrival(args, np.array(database_hash), query, location=location)
    print("-------------------------")
    print("generating retrieval samples")
    for i in range(len(ids)):
        current_is = ids[i]
        category = cat[int(database_labels[current_is][0])]
        if args.dataset == "MNIST":
            img_orl = Image.fromarray(imgs_database[current_is].numpy())
        elif args.dataset == "cifar10":
            img_orl = Image.fromarray(imgs_database[current_is])
        else:
            img_orl = Image.open(imgs_database[current_is]).convert('RGB')
        img_orl = img_orl.resize([224, 224], resample=Image.BILINEAR)
        cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, [i, category, location])
        img_orl.save(f'vis_pp/orl_{i}_{category}.png')
        slot_image = np.array(Image.open(f'vis_pp/mask_{i}_{category}.png'), dtype=np.uint8)
        heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl, slot_image, 'jet')
        heatmap_on_image.save("vis_pp/" + f'jet_{i}_{category}.png')


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = False
    main()