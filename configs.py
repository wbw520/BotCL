# Code for concept

import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of cpt")
parser.add_argument('--dataset', type=str, default="CUB200")
parser.add_argument('--dataset_dir', type=str, default="/media/wbw/a7f02863-b441-49d0-b546-6ef6fefbbc7e")
parser.add_argument('--output_dir', type=str, default="saved_model")
# ========================= Model Configs ==========================
parser.add_argument('--num_classes', default=50, type=int, help='category for classification')
parser.add_argument('--num_cpt', default=20, type=int, help='number of the concept')
parser.add_argument('--base_model', default="resnet18", type=str)
parser.add_argument('--img_size', default=224, help='size for input image')
parser.add_argument('--pre_train', default=True, type=bool,
                    help='whether pre-train the model')
parser.add_argument('--aug', default=True, type=bool,
                    help='whether use augmentation')
parser.add_argument('--act_type', default="sigmoid", help='the activation for the slot attention')
parser.add_argument('--num_retrieval', default=20, help='number of the top retrieval images')
parser.add_argument('--weight_att', default=False, help='using fc weight for att visualization')
parser.add_argument('--cpt_activation', default="att", help='the type to form cpt activation')
parser.add_argument('--feature_size', default=7, help='size of the feature from backbone')
parser.add_argument('--process', default=False, help='whether process for h5py file')
parser.add_argument('--layer', default=2, help='layers for fc, default as one')
# ========================= Training Configs ==========================
parser.add_argument('--weak_supervision_bias', type=float, default=1, help='weight fot the weak supervision branch')
parser.add_argument('--att_bias', type=float, default=0.1, help='used to prevent overflow, default as 0.1')
parser.add_argument('--quantity_bias', type=float, default=0.5, help='force each concept to be binary')
parser.add_argument('--distinctiveness_bias', type=float, default=1, help='refer to paper')
parser.add_argument('--consistence_bias', type=float, default=1, help='refer to paper')
# ========================= Learning Configs ==========================
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epoch', default=20, type=int)
parser.add_argument('--lr_drop', default=10, type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
# ========================= Machine Configs ==========================
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
# ========================= Demo Configs ==========================
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--use_weight', default=False, help='whether use fc weight for the generation of attention mask')
parser.add_argument('--top_samples', default=50, type=int, help='top n activated samples')
# parser.add_argument('--demo_cls', default="n01498041", type=str)
parser.add_argument('--fre', default=3, type=int, help='frequent of show results during training')

