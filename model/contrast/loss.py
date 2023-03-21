import torch
from torch.autograd import Variable
import torch.nn.functional as F


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


def pairwise_similarity_label(label):
    pair_label = F.cosine_similarity(label[:, None, :], label[None, :, :], dim=2)
    return pair_label


def soft_similarity(features, label):
    s_loss = torch.abs(torch.sigmoid(features) - label)
    return torch.mean(s_loss)


def hard_similarity(dot_product, similarity):
    exp_product = torch.exp(dot_product)
    exp_loss = (torch.log(1 + exp_product) - similarity * dot_product)
    mask_positive = similarity > 0
    mask_negative = similarity <= 0
    S1 = torch.sum(mask_positive.float())
    S0 = torch.sum(mask_negative.float())
    S = S0 + S1

    exp_loss[similarity > 0] = exp_loss[similarity > 0] * (S / S1)
    exp_loss[similarity <= 0] = exp_loss[similarity <= 0] * (S / S0)
    loss = torch.mean(exp_loss)

    return loss


def pairwise_loss2(feature1, feature2, label, sigmoid_param=1.):
    label_similarity = pairwise_similarity_label(label)
    features_dis = sigmoid_param * torch.mm(feature1, feature2.t())

    label_similarity = label_similarity.reshape(-1)
    features_dis = features_dis.reshape(-1)
    hard_index_1 = label_similarity == 0
    hard_index_2 = label_similarity == 1
    hard_index = hard_index_1 | hard_index_2
    soft_index = ~ hard_index
    similarity_hard = label_similarity[hard_index]
    similarity_soft = label_similarity[soft_index]

    features_dis_hard = features_dis[hard_index]
    features_dis_soft = features_dis[soft_index]

    hard_loss = hard_similarity(features_dis_hard, similarity_hard)
    soft_loss = soft_similarity(features_dis_soft, similarity_soft)

    return (hard_loss + soft_loss) / 2


def quantization_loss(cpt):
    q_loss = torch.mean((torch.abs(cpt)-1.0)**2)
    return q_loss


def get_retrieval_loss(args, y, label, num_cls, device):
    b = label.shape[0]
    if args.dataset != "matplot":
        label = label.unsqueeze(-1)
        label = torch.zeros(b, num_cls).to(device).scatter(1, label, 1)
    similarity_loss = pairwise_loss(y, y, label, label, sigmoid_param=10. / 32)
    # similarity_loss = pairwise_loss2(y, y, label.float(), sigmoid_param=10. / 32)
    q_loss = quantization_loss(y)
    return similarity_loss, q_loss


def batch_cpt_discriminate(data, att):
    b1, c, d1 = data.shape
    record = []
    for i in range(c):
        current_f = data[:, i, :]
        current_att = att.sum(-1)[:, i]
        indices = current_att > current_att.mean()
        b, d = current_f[indices].shape
        current_f = current_f[indices]
        record.append(torch.mean(current_f, dim=0, keepdim=True))
    record = torch.cat(record, dim=0)
    sim = F.cosine_similarity(record[None, :, :], record[:, None, :], dim=-1)
    return sim.mean()


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
        b, d = current_up[indices].shape
        need = current_up[indices]
        consistence_loss += F.cosine_similarity(need[None, :, :], need[:, None, :], dim=-1).mean()
    return consistence_loss/cpt


def att_area_loss(att):
    slot_loss = torch.sum(att, (0, 1, 2)) / att.size(0) / att.size(1) / att.size(2)
    return torch.pow(slot_loss, 1)
