from model.retrieval.model_main import MainModel
from configs import parser
import torch
import os
from PIL import Image
import numpy as np
from utils.tools import apply_colormap_on_image
from loaders.get_loader import load_all_imgs, get_transformations_synthetic
import h5py
import shutil


shutil.rmtree('vis/', ignore_errors=True)
shutil.rmtree('vis_pp/', ignore_errors=True)
os.makedirs('vis/', exist_ok=True)
os.makedirs('vis_pp/', exist_ok=True)
np.set_printoptions(suppress=True)


def main():
    # load all imgs
    imgs_database, labels_database, imgs_val, labels_val = load_all_imgs(args)
    # print("All category:")
    # print(cat)
    transform = get_transformations_synthetic()

    # load model and weights
    model = MainModel(args, vis=True)
    device = torch.device("cuda:0")
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir,
            f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" +
    f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}_version3.pt"), map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    img = Image.open(os.path.join("../loaders", "matplob", "raw", "5684.jpg")).convert('RGB')
    img = img.resize([224, 224], resample=Image.BILINEAR)
    img.save(f'vis/origin.png')
    cpt, pred, att, update = model(transform(img).unsqueeze(0).to(device), None, None)
    print("-------------------------")
    pp = torch.argmax(pred, dim=-1)
    print(cpt)
    print(pp)

    w = model.state_dict()["cls.weight"][pp]
    w_numpy = np.around(torch.tanh(w).cpu().detach().numpy(), 4)

    print("--------weight---------")
    print(w_numpy)

    # get retrieval cases
    f1 = h5py.File(f"data_map/{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_{args.cpt_activation}.hdf5", 'r')
    database_hash = f1["database_hash"]
    database_labels = f1["database_labels"]
    test_hash = f1["test_hash"]
    test_labels = f1["test_labels"]

    print("-------------------------")
    print("generating concept samples")

    for j in range(args.num_cpt):
        root = 'vis_pp/' + "cpt" + str(j) + "/"
        os.makedirs(root, exist_ok=True)
        selected = np.array(test_hash)[:, j]
        ids = np.argsort(-selected, axis=0)
        idx = ids[:args.top_samples]
        for i in range(len(idx)):
            current_is = idx[i]
            # category = cat[int(test_labels[current_is][0])]
            category = "-"
            img_orl = Image.open(imgs_val[current_is]).convert('RGB')
            img_orl = img_orl.resize([224, 224], resample=Image.BILINEAR)
            cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, [i, category, j])
            img_orl.save(root + f'/orl_{i}_{category}.png')
            slot_image = np.array(Image.open(root + f'mask_{i}_{category}.png'), dtype=np.uint8)
            heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl, slot_image, 'jet')
            heatmap_on_image.save(root + f'jet_{i}_{category}.png')


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = False
    main()