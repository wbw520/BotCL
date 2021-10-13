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
    similarity_loss = pairwise_loss(y, y, label, label, sigmoid_param=10. / 32)
    q_loss = quantization_loss(y)
    return similarity_loss, q_loss


def batch_cpt_discriminate(data):
    data = data.mean(0)
    sim = ((data[None, :, :] - data[:, None, :]) ** 2).sum(-1)
    return torch.tanh(sim).mean()


def att_binary(att):
    att = (att - 0.5) * 2
    return torch.mean((torch.abs(att)-1.0)**2)


def att_discriminate(att):
    b, cpt, spatial = att.size()
    att_mean = torch.sum(att, dim=-1)
    dis_loss = 0.0
    for i in range(b):
        current_mean = att_mean[i].mean()
        indices = att_mean[i] > current_mean
        need = att[i][indices]
        dis_loss += torch.tanh(((need[None, :, :] - need[:, None, :]) ** 2).sum(-1)).mean()
    return dis_loss/b


def att_consistence(update, att):
    b, cpt, spatial = att.size()
    consistence_loss = 0.0
    for i in range(cpt):
        current_up = update[:, i, :]
        current_att = att[:, i, :].sum(-1)
        indices = current_att > current_att.mean()
        need = current_up[indices]
        consistence_loss += torch.tanh(((need[None, :, :] - need[:, None, :]) ** 2).sum(-1)).mean()
    return consistence_loss/cpt


def att_area_loss(att):
    slot_loss = torch.sum(att, (0, 1, 2)) / att.size(0) / att.size(1) / att.size(2)
    return torch.pow(slot_loss, 1)
