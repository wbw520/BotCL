from torch import nn
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np


class ScouterAttention(nn.Module):
    def __init__(self, args, dim, num_concept, iters=3, eps=1e-8, vis=False, power=1, to_k_layer=3):
        super().__init__()
        self.args = args
        self.num_slots = num_concept
        self.slot_mode = args.act_type
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        # random seed init
        slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))
        mu = slots_mu.expand(1, self.num_slots, -1)
        sigma = slots_sigma.expand(1, self.num_slots, -1)
        self.initial_slots = nn.Parameter(torch.normal(mu, sigma))

        # K layer init
        to_k_layer_list = [nn.Linear(dim, dim)]
        for to_k_layer_id in range(1, to_k_layer):
            to_k_layer_list.append(nn.ReLU(inplace=True))
            to_k_layer_list.append(nn.Linear(dim, dim))
        self.to_k = nn.Sequential(
            *to_k_layer_list
        )

        self.vis = vis
        self.power = power

    def forward(self, inputs_pe, inputs, weight=None, things=None):
        b, n, d = inputs_pe.shape
        slots = self.initial_slots.expand(b, -1, -1)
        k, v = self.to_k(inputs_pe), inputs_pe
        for _ in range(self.iters):
            q = slots

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            dots = torch.div(dots, dots.sum(2).expand_as(dots.permute([2, 0, 1])).permute([1, 2, 0])) * \
                   dots.sum(2).sum(1).expand_as(dots.permute([1, 2, 0])).permute([2, 0, 1])
            if self.slot_mode == "sigmoid":
                attn = torch.sigmoid(dots)
            elif self.slot_mode == "softmax":
                attn1 = dots.softmax(dim=1)
                attn2 = dots.sigmoid()
                attn = attn1 * attn2
            elif self.slot_mode == "no_act":
                attn = dots
            else:
                raise RuntimeError(f"unsupported input to tensor dot, got slot mode={self.slot_mode}")

            updates = torch.einsum('bjd,bij->bid', inputs, attn)
            updates = updates / inputs.size(2)

        if self.vis:
            slots_vis_raw = attn.clone()
            vis(slots_vis_raw, "vis", self.args.feature_size, weight, things)
        return updates, attn


def vis(slots_vis_raw, loc, size, weight=None, things=None):
    b = slots_vis_raw.size()[0]
    for i in range(b):
        slots_vis = slots_vis_raw[i]
        if weight is not None:
            slots_vis = slots_vis * weight.unsqueeze(-1)
        slots_vis = ((slots_vis - slots_vis.min()) / (slots_vis.max() - slots_vis.min()) * 255.).reshape(
            slots_vis.shape[:1] + (int(size), int(size)))

        slots_vis = (slots_vis.cpu().detach().numpy()).astype(np.uint8)
        for id, image in enumerate(slots_vis):
            image = Image.fromarray(image, mode='L').resize([224, 224], resample=Image.BILINEAR)
            if things is not None:
                order, category, cpt_num = things
                if id == cpt_num:
                    image.save(f'vis_pp/mask_{order}_{category}.png')
                    break
                else:
                    continue
            image.save(f'{loc}/{i}_slot_{id:d}.png')