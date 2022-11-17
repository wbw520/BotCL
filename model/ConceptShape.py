import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
from utils.tools import cal_acc
import os


class ConceptShapModel(nn.Module):
    def __init__(self, n_concepts, n_features, B_threshold):
        super(ConceptShapModel, self).__init__()
        self.n_concepts = n_concepts
        self.n_features = n_features
        self.B_threshold = B_threshold

        self.g = nn.Sequential(
            nn.Linear(n_concepts, 500),
            nn.ReLU(),
            nn.Linear(500, n_features)
        )
        self.C = nn.Linear(n_features, n_concepts, bias=False)

    def forward(self, x):
        g_vc_flat = self.g(x)
        return g_vc_flat

    def vc(self, x):
        vc_flat = self.C(x)
        vc_flat = nn.Threshold(self.B_threshold, 0, inplace=False)(vc_flat)
        vc_flat = F.normalize(vc_flat, p=2.0, dim=1)
        return vc_flat


class ConceptShap:
    def __init__(self,
        args,
        model,
        train_loader,
        val_loader,
        shape=(224, 224),
        B_threshold = 0.2,
        cshap_optim_lr=0.01,
        lambda_1=0.1,
        lambda_2=0.1,
        MC_shapely_samples=20,

        batch_size=8,
        n_concept_examples=40,
        ):
        """Initialization of the algorithm"""
        ##############################
        self.args = args
        self.save_path = args.output_dir
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.shape = shape
        self.n_concepts = args.num_cpt
        self.B_threshold = B_threshold
        self.cshap_optim_lr = cshap_optim_lr
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.MC_shapely_samples = MC_shapely_samples
        self.batch_size = batch_size
        self.device = args.device
        self.n_concept_examples = n_concept_examples
        self.K = None

        self.cshap_model = ConceptShapModel(self.n_concepts, 512, self.B_threshold).to(self.device)
        self.model.to(self.device)

    def eval_forward(self, x):
        pred, activation = self.model(x)
        a_shape = activation.shape
        a_flatten = activation.permute(0, 2, 3, 1).flatten(0, 2)
        vc_flat = self.cshap_model.vc(a_flatten)
        g_vc_flat = self.cshap_model(vc_flat)
        print(g_vc_flat.shape)
        g_vc = g_vc_flat.unflatten(0, [a_shape[0], a_shape[2], a_shape[3]]).permute(0, 3, 1, 2)
        print(g_vc.shape)
        sdfds()
        g_next = F.adaptive_max_pool2d(g_vc, 1).squeeze(-1).squeeze(-1)
        out = self.model.fc(g_next)
        return out, activation

    def learn_concepts(self, lr=None, epochs=200):
        if lr is None: lr = self.cshap_optim_lr
        self.cshap_model.train()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.K = 10
        # init optimization
        optimizer = torch.optim.SGD(self.cshap_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=0.00001, mode='min')
        # iterate train
        for epoch in range(epochs):
            accs = []
            losses = []
            print("----------" + str(epoch) + "-----------")
            for batch_idx, (data, target) in enumerate(self.train_loader):
                x, y = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                # model loss
                logits, act = self.eval_forward(x)
                pred = F.sigmoid(logits)
                loss = F.binary_cross_entropy(pred, y.float())
                acc = torch.eq(pred.round(), y).sum().float().item() / pred.shape[0] / pred.shape[1]
                accs.append(acc)
                # R(c)

                a_flat = act.permute(0, 2, 3, 1).flatten(0, 2)
                a_norm = F.normalize(a_flat, p=2.0, dim=1)
                c_norm = F.normalize(self.cshap_model.C.weight, p=2.0, dim=1)
                loss_proj = torch.sum(c_norm @ a_flat.T)/(self.K*self.n_concepts) # modified to avoid non unitary concept vectors
                loss_novelty = torch.tril((c_norm @ c_norm.T), diagonal=-1).sum()/(self.n_concepts*(self.n_concepts-1))
                loss -= self.lambda_1 * loss_proj - self.lambda_2 * loss_novelty

                losses.append(loss.detach().cpu().numpy())
                # backward
                loss.backward()
                optimizer.step()

            print("acc: ", np.array(accs).mean())
            print("loss: ", np.array(losses).mean())
            scheduler.step(np.array(losses).mean())

        torch.save(self.cshap_model.state_dict(), os.path.join(self.args.output_dir,
                                                    f"{self.args.dataset}_{self.args.base_model}_cls{self.args.num_classes}_" + f"cpt{self.args.num_cpt if not self.args.pre_train else ''}_" +
                                                    "ConceptShape.pt"))