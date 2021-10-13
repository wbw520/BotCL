# Code for concept
# Bowen Wang
# bowen.wang@is.ids.osaka-u.ac.jp

import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of cpt")
parser.add_argument('--dataset', type=str, default="CUB200")
parser.add_argument('--dataset_dir', type=str, default="/media/wbw/a7f02863-b441-49d0-b546-6ef6fefbbc7e")
parser.add_argument('--output_dir', type=str, default="saved_model")
# ========================= Model Configs ==========================
parser.add_argument('--num_classes', default=50, help='category for classification')
parser.add_argument('--num_cpt', default=25, help='number of the concept')
parser.add_argument('--base_model', default="resnet18", type=str)
parser.add_argument('--img_size', default=224, help='size for input image')
parser.add_argument('--pre_train', default=True, type=bool,
                    help='whether use ImageNet pre-train parameter for backbone')
parser.add_argument('--act_type', default="sigmoid", help='the activation for the slot attention')
parser.add_argument('--num_retrieval', default=50, help='number of the top retrieval images')
parser.add_argument('--weight_att', default=False, help='using fc weight for att visualization')
parser.add_argument('--cpt_activation', default="att", help='the type to form cpt activation')
parser.add_argument('--feature_size', default=7, help='size of the feature from backbone')
parser.add_argument('--process', default=False, help='whether process for h5py file')
# ========================= Training Configs ==========================

# ========================= Learning Configs ==========================
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epoch', default=40, type=int)
parser.add_argument('--lr_drop', default=20, type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
# ========================= Machine Configs ==========================
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--device', default='cuda:0',
                    help='device to use for training / testing')
