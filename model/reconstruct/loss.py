import torch
from torch.autograd import Variable


def pairwise_loss(outputs1, outputs2, label1, label2, sigmoid_param=1.):
    similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())
    exp_product = torch.exp(dot_product)

    exp_loss = (torch.log(1 + exp_product) - similarity * dot_product)
    mask_positive = similarity > 0
    mask_negative = similarity <= 0
    S1 = torch.sum(mask_positive.float())
    S0 = torch.sum(mask_negative.float())
    S = S0+S1

    exp_loss[similarity > 0] = exp_loss[similarity > 0] * (S / S1)
    exp_loss[similarity <= 0] = exp_loss[similarity <= 0] * (S / S0)

    loss = torch.mean(exp_loss)

    return loss


def quantization_loss(cpt):
    q_loss = torch.mean((torch.abs(cpt)-1.0)**2)
    return q_loss


def get_retrieval_loss(y, label, num_cls, device):
    b = label.shape[0]
    label = label.unsqueeze(-1)
    label = torch.zeros(b, num_cls).to(device).scatter(1, label, 1)
    similarity_loss = pairwise_loss(y, y, label, label, sigmoid_param=1)
    q_loss = quantization_loss(y)
    re_loss = similarity_loss + 0.1 * q_loss
    return re_loss
