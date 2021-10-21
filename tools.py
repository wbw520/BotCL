import numpy as np
import torch
from PIL import Image


def cal_acc(preds, labels):
    with torch.no_grad():
        pred = preds.argmax(dim=-1)
        acc = torch.eq(pred, labels).sum().float().item() / labels.size(0)
        return acc


def mean_average_precision(args, database_hash, test_hash, database_labels, test_labels):  # R = 1000
    # binary the hash code
    R = args.num_retrieval
    # database_hash[database_hash < T] = -1
    # database_hash[database_hash >= T] = 1
    # test_hash[test_hash < T] = -1
    # test_hash[test_hash >= T] = 1

    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)
    ids = np.argsort(-sim, axis=0)

    APx = []
    Recall = []

    for i in range(query_num):  # for i=0
        if i % 2000 == 0:
            print(str(i) + "/" + str(query_num))
        label = test_labels[i, :]  # the first test labels
        # label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0

        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)  #

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float(all_num)
        Recall.append(r)

    return np.mean(np.array(APx)), np.mean(np.array(Recall)), APx


def predict_hash_code(args, model, data_loader, device):
    model.eval()
    is_start = True
    accs = 0
    L = len(data_loader)

    for batch_idx, (data, label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device)
        if not args.pre_train:
            cpt, pred, att, update = model(data)
            acc = cal_acc(pred, label)
            accs += acc
        else:
            cpt = model(data)
        if is_start:
            all_output = cpt.cpu().detach().float()
            all_label = label.unsqueeze(-1).cpu().detach().float()
            is_start = False
        else:
            all_output = torch.cat((all_output, cpt.cpu().detach().float()), 0)
            all_label = torch.cat((all_label, label.unsqueeze(-1).cpu().detach().float()), 0)

    return all_output.numpy().astype("float32"), all_label.numpy().astype("float32"), round(accs/L, 4)


def test_MAP(args, model, database_loader, test_loader, device):
    print('Waiting for generate the hash code from database')
    database_hash, database_labels, database_acc = predict_hash_code(args, model, database_loader, device)
    print('Waiting for generate the hash code from test set')
    test_hash, test_labels, test_acc = predict_hash_code(args, model, test_loader, device)
    print('Calculate MAP.....')
    MAP, R, APx = mean_average_precision(database_hash, test_hash, database_labels, test_labels)

    return MAP, test_acc


def fix_parameter(model, name_fix, mode="open"):
    """
    fix parameter for model training
    """
    for name, param in model.named_parameters():
        for i in range(len(name_fix)):
            if mode != "fix":
                if name_fix[i] not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    break
            else:
                if name_fix[i] in name:
                    param.requires_grad = False


def print_param(model):
    # show name of parameter could be trained in model
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


def for_retrival(args, database_hash, test_hash, location):
    R = args.num_retrieval

    # database_hash[database_hash < T] = -1
    # database_hash[database_hash >= T] = 1
    # test_hash[test_hash < T] = -1
    # test_hash[test_hash >= T] = 1
    sim = np.matmul(database_hash[:, location:location+1], test_hash[:, location:location+1].T)
    ids = np.argsort(-sim, axis=0)
    idx = ids[:, 0]
    ids = idx[:R]
    return ids


def attention_estimation(data, label, model, transform, device):
    selected_class = "Yellow_headed_Blackbird"
    contains = []
    for i in range(len(data)):
        if selected_class in data[i]:
            contains.append(data[i])

    attention_record = []
    for i in range(len(contains)):
        img_orl = Image.open(contains[i]).convert('RGB')
        img_orl = img_orl.resize([224, 224], resample=Image.BILINEAR)
        cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, None)
        attention_record.append(att.sum(-1).squeeze(0).cpu().detach().numpy())
    return np.array(attention_record)
