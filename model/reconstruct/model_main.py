import torch.nn as nn
import torch
from model.reconstruct.slots import ScouterAttention
from model.reconstruct.position_encode import build_position_encoding
import torch.nn.functional as F


class ConceptAutoencoder(nn.Module):
    def __init__(self, args, num_concepts, vis=False):
        super(ConceptAutoencoder, self).__init__()
        hidden_dim = 32
        self.args = args
        self.num_concepts = num_concepts
        self.conv1 = nn.Conv2d(1, 16, (3, 3), stride=2, padding=1)  # b, 16, 10, 10
        self.conv2 = nn.Conv2d(16, hidden_dim, (3, 3), stride=2, padding=1)  # b, 8, 3, 3
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(num_concepts, 400)
        self.fc2 = nn.Linear(400, 28 * 28)
        self.tan = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.vis = vis
        self.scale = 1
        self.activation = nn.Tanh()
        self.position_emb = build_position_encoding('sine', hidden_dim=hidden_dim)
        self.slots = ScouterAttention(hidden_dim, num_concepts, vis=self.vis)
        self.aggregate = Aggregate(args, num_concepts)

    def forward(self, x, loc=None, index=None):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        pe = self.position_emb(x)
        x_pe = x + pe
        b, n, r, c = x.shape
        x = x.reshape((b, n, -1)).permute((0, 2, 1))
        x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
        updates, attn = self.slots(x_pe, x, loc, index)
        cpt_activation = attn
        attn_cls = self.scale * torch.sum(cpt_activation, dim=-1)

        x = attn_cls.reshape(b, -1)
        cpt = self.activation(x)
        x = cpt
        if self.args.deactivate != -1:
            x[0][self.args.deactivate-1] = 0
        pred = self.aggregate(x)
        x = self.relu(self.fc1(x))
        x = self.tan(self.fc2(x))
        return (cpt - 0.5) * 2, pred, x, attn, updates


class Aggregate(nn.Module):
    def __init__(self, args, num_concepts):
        super(Aggregate, self).__init__()
        self.args = args
        if args.layer != 1:
            self.fc1 = nn.Linear(num_concepts, num_concepts)
        self.fc2 = nn.Linear(num_concepts, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.args.layer != 1:
            x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MNISTSimple(nn.Module):
    def __init__(self):
        super(MNISTSimple, self).__init__()
        hidden_dim = 32
        self.conv1 = nn.Conv2d(1, 10, (5, 5), stride=2, padding=1)  # b, 16, 10, 10
        self.conv2 = nn.Conv2d(10, hidden_dim, (5, 5), stride=2, padding=1)  # b, 8, 3, 3
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        f = x
        x = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
        pred = self.fc(x)
        return pred, f

# if __name__ == '__main__':
#     model = ConceptAutoencoder(num_concepts=10)
#     inp = torch.rand((2, 1, 28, 28))
#     pred, out, att_loss = model(inp)
#     print(pred.shape)
#     print(out.shape)