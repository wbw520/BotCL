import os
import torch
from configs import parser
import torch.nn.functional as F
from model.retrieval.model_main import MainModel
import pickle
import json
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from PIL import Image


os.makedirs('pickle_files/', exist_ok=True)


@torch.no_grad()
def evaluation(args, model, loader, device):
    model.eval()

    for batch_idx, (data, img_name) in enumerate(loader):
        print(batch_idx)
        data = data.to(device, dtype=torch.float32)
        if not args.pre_train:
            cpt, pred, att, update = model(data)
            save_f = att
        else:
            pred, features = model(data)
            save_f = features

        if not args.pre_train:
            b = save_f.size()[0]
            f_record = []
            for i in range(b):
                slots_vis = save_f[i]
                a = (slots_vis - slots_vis.min()) / (slots_vis.max() - slots_vis.min())
                slots_vis = (slots_vis - slots_vis.min()) / (slots_vis.max() - slots_vis.min()).reshape(
                    slots_vis.shape[:1] + (7, 7))
                f_record.append(slots_vis.unsqueeze(0))
            save_f = torch.cat(f_record, dim=0)

        save_f = save_f.cpu().detach().float()

        for i in range(len(img_name)):
            save_file = {}
            save_file.update({"imgs": save_f[i]})
            names = img_name[i].split("/")
            new_name = names[0] + "-" + names[1]
            print(new_name)
            with open('pickle_files/' + new_name + "_" + f"{'use_slot' if not args.pre_train else 'no_slot'}_activation.pkl", 'wb') as f:
                pickle.dump(save_file, f, pickle.HIGHEST_PROTOCOL)


class Broden(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        with open("/home/wangbowen/PycharmProjects/broden/images_need_cal.json", "r") as f:
            row_data = json.load(f)
        self.all_data = row_data["imgs"]
        self.transform = transform

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item_id):
        image_name = self.all_data[item_id]
        image_root = "/home/wangbowen/DATA/Broden/broden1_227/images/" + image_name
        image = Image.open(image_root).convert('RGB')
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_name


def get_transformations():
    norm_value = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    aug_list = [
                transforms.Resize((224, 224), Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(norm_value[0], norm_value[1])
                ]
    return transforms.Compose(aug_list)


def get_loader():
    trans = get_transformations()
    dataset = Broden(trans)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)
    return loader


def main():
    # load model and weights
    model = MainModel(args, vis=False)
    device = torch.device("cuda:1")
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir,
                f"{args.dataset}_{args.base_model}_cls{args.num_classes}_" + f"cpt{args.num_cpt if not args.pre_train else ''}_" +
                f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"),
                            map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    loader = get_loader()
    evaluation(args, model, loader, device)


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = False
    main()