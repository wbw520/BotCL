from loaders.get_loader import loader_generation
from configs import parser
import sklearn.cluster as cluster
from sklearn.decomposition import PCA
from termcolor import colored
import torch.nn.functional as F
import torch
import os
import pickle
import numpy as np
from model.reconstruct.model_main import MNISTSimple
from model.retrieval.model_main import MainModel
from utils.record import AverageMeter, ProgressMeter, show
from utils.tools import cal_acc, predict_hash_code, mean_average_precision
import torch.nn as nn


def get_model():
    if args.dataset == "MNIST":
        model = MNISTSimple()
    else:
        model = MainModel(args)

    return model


class FC(nn.Module):
    def __init__(self, args, hidden_dim):
        super(FC, self).__init__()
        self.fc = nn.Linear(hidden_dim, args.num_classes)
        self.temp = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = torch.tanh(x * torch.abs(self.temp))
        pred = self.fc(x)
        return pred


def engine_train(args, model, device, loader, optimizer, epoch):
    cls_loss = AverageMeter('Cls', ':.4')
    pred_acces = AverageMeter('Acc', ':.4')
    show_items = [pred_acces, cls_loss]
    progress = ProgressMeter(len(loader),
                             show_items,
                             prefix="Epoch: [{}]".format(epoch))

    model.train()
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.int64)
        pred, features = model(data)

        if args.dataset != "matplot":
            pred = F.log_softmax(pred, dim=-1)
            loss_pred = F.nll_loss(pred, label)
            acc = cal_acc(pred, label, False)
        else:
            pred = F.sigmoid(pred)
            loss_pred = F.binary_cross_entropy(pred, label.float())
            acc = torch.eq(pred.round(), label).sum().float().item() / pred.shape[0] / pred.shape[1]

        pred_acces.update(acc)
        cls_loss.update(loss_pred)
        loss_total = loss_pred

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            progress.display(batch_idx)


@torch.no_grad()
def test(args, model, test_loader, device):
    model.eval()
    record = 0.0
    for batch_idx, (data, label) in enumerate(test_loader):
        data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.int64)
        pred, features = model(data)
        if args.dataset != "matplot":
            pred = F.log_softmax(pred, dim=-1)
            acc = cal_acc(pred, label, False)
        else:
            pred = F.sigmoid(pred)
            acc = torch.eq(pred.round(), label).sum().float().item() / pred.shape[0] / pred.shape[1]
        record += acc
    ACC = record/len(test_loader)
    print("ACC:", record/len(test_loader))
    return ACC


class PreTraining():
    def __init__(self, model):
        self.model = model
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=args.lr)
        self.model.to(device)

    def main(self):
        acc_max = 0

        for i in range(args.epoch):
            print(colored('Epoch %d/%d' % (i + 1, args.epoch), 'yellow'))
            print(colored('-' * 15, 'yellow'))
            engine_train(args, self.model, device, train_loader1, self.opt, i)

            print("start evaluation")
            acc = test(args, self.model, val_loader, device)

            if acc > acc_max:
                acc_max = acc
                print("get better result, save current model.")
                torch.save(self.model.state_dict(), os.path.join(args.output_dir,
                                                            f"{args.dataset}_{args.base_model}_cls{args.num_classes}_extra.pt"))


class CalCenter():
    def __init__(self, model):
        self.model = model
        checkpoint = torch.load(os.path.join(args.output_dir, f"{args.dataset}_{args.base_model}_cls{args.num_classes}_extra.pt"),
                                map_location="cuda:0")
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()
        self.model.to(device)

    def kmeans(self, acts):
        km = cluster.KMeans(n_clusters, random_state=2)
        d = km.fit(acts)
        centers = km.cluster_centers_
        with open("pickle_file/" + args.dataset + str(n_clusters) + "_kmeans.pickle", "wb") as file:
            pickle.dump({"center": centers}, file)
        file.close()
        print("kmeans finished")

    def pca(self, acts_train):
        pca = PCA(n_components=n_clusters)
        x_train = pca.fit(acts_train)
        axis = pca.components_
        with open("pickle_file/" + args.dataset + str(n_clusters) + "_pca.pickle", "wb") as file:
            pickle.dump({"axis": axis}, file)
        file.close()
        print("pca finished")

    def get_acts(self, loader):
        output = []
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.int64)
            _, features = self.model(data)
            features = features.permute([0, 2, 3, 1]).reshape(-1, size * size, dim)
            output.append(features.cpu().detach().numpy())
        output = np.concatenate(output, 0)
        output = output.reshape(-1, dim)
        return output

    def main(self):
        acts = self.get_acts(train_loader2)
        if method == "kmeans":
            self.kmeans(acts)
        elif method == "pca":
            self.pca(acts)


class Train():
    def __init__(self, model, fc):
        self.model = model
        checkpoint = torch.load(
            os.path.join(args.output_dir, f"{args.dataset}_{args.base_model}_cls{args.num_classes}_extra.pt"),
            map_location="cuda:0")
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()
        self.model.to(device)

        self.fc = fc
        params = [p for p in self.fc.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=args.lr)
        self.fc.to(device)

    def train_(self):
        if method == "kmeans":
            with open("pickle_file/" + args.dataset + str(n_clusters) + "_kmeans.pickle", "rb") as file:
                center = pickle.load(file)["center"]
        else:
            with open("pickle_file/" + args.dataset + str(n_clusters) + "_pca.pickle", "rb") as file:
                center = pickle.load(file)["axis"]
        center = torch.from_numpy(center).to(device)
        acc_max = 0

        for i in range(args.epoch):
            print("epoch: ", str(i))
            self.fc.train()
            i = 0
            for item in [train_loader1, val_loader]:
                acc_ = 0
                for batch_idx, (data, label) in enumerate(item):
                    data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.int64)
                    _, features = self.model(data)
                    features = features.permute([0, 2, 3, 1]).reshape(-1, size * size, dim)
                    if method == "kmeans":
                        active = self.kmeans_d(center, features)
                    else:
                        active = self.pca_d(center, features)
                    pred = self.fc(active)

                    if args.dataset != "matplot":
                        pred = F.log_softmax(pred, dim=-1)
                        loss_pred = F.nll_loss(pred, label)
                        acc = cal_acc(pred, label, False)
                    else:
                        pred = F.sigmoid(pred)
                        loss_pred = F.binary_cross_entropy(pred, label.float())
                        acc = torch.eq(pred.round(), label).sum().float().item() / pred.shape[0] / pred.shape[1]

                    loss_total = loss_pred
                    self.opt.zero_grad()
                    loss_total.backward()
                    self.opt.step()
                    acc_ += acc
                c_acc = acc_ / len(item)
                if i == 1:
                    if c_acc > acc_max:
                        acc_max = c_acc
                        print(c_acc)
                        print("get better result, save current model.")
                        torch.save(self.model.state_dict(), os.path.join(args.output_dir,
                        f"{args.dataset}_{args.base_model}_cls{args.num_classes}_extra.pt"))
                i += 1

    def kmeans_d(self, center, features):
        d = ((features[:, :, None, :] - center[None, None, :, :]) ** 2).mean(-1)
        d = torch.exp(-d)
        d = torch.div(d, d.sum(-1).expand_as(d.permute([2, 0, 1])).permute([1, 2, 0]))
        active = d.sum(1)
        return active

    def pca_d(self, axis, features):
        n, l, c = features.shape
        mean = torch.mean(features, dim=(0, 1)).expand(n, l, c)
        var = torch.var(features, dim=(0, 1)).expand(n, l, c)
        f = (features - mean) / var
        d = torch.abs((f[:, :, None, :] * axis[None, None, :, :]).sum(-1))
        d = torch.div(d, d.sum(-1).expand_as(d.permute([2, 0, 1])).permute([1, 2, 0]))
        active = d.sum(1)
        return active


if __name__ == '__main__':
    os.makedirs('pickle_file/', exist_ok=True)
    args = parser.parse_args()
    args.dataset = "ImageNet"
    method = "pca"
    args.pre_train = True
    args.epoch = 50
    n_clusters = 50
    args.num_classes = 200
    args.base_model = "resnet18"
    dim = 512
    size = 7
    mode = "train"
    model_bone = get_model()
    device = torch.device(args.device)

    # CUDNN
    torch.backends.cudnn.benchmark = True
    train_loader1, train_loader2, val_loader = loader_generation(args)

    if mode == "pre_training":
        PreTraining(model_bone).main()
    elif mode == "cal_center":
        CalCenter(model_bone).main()
    elif mode == "train":
        fc_ = FC(args, n_clusters)
        Train(model_bone, fc_).train_()
