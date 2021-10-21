import torch
import torch.nn.functional as F
from loss import get_retrieval_loss, batch_cpt_discriminate, att_consistence, att_discriminate, att_binary, \
    att_area_loss
from utils import AverageMeter, ProgressMeter, show
from tools import cal_acc, predict_hash_code, mean_average_precision


def train(args, model, device, loader, optimizer, epoch):
    retri_losses = AverageMeter('Retri_loss Loss', ':.4')
    att_losses = AverageMeter('Att Loss', ':.4')
    q_losses = AverageMeter('Q_loss', ':.4')
    batch_dis_losses = AverageMeter('Dis_loss_batch', ':.4')
    consistence_losses = AverageMeter('Consistence_loss', ':.4')
    att_binary_losses = AverageMeter('Att_binary', ':.4')
    att_dis_losses = AverageMeter('Att_dis', ':.4')
    pred_acces = AverageMeter('Acc', ':.4')
    if not args.pre_train:
        show_items = [retri_losses, q_losses, att_losses, pred_acces, batch_dis_losses, consistence_losses,
                      att_binary_losses, att_dis_losses]
    else:
        show_items = [retri_losses, q_losses]
    progress = ProgressMeter(len(loader),
                             show_items,
                             prefix="Epoch: [{}]".format(epoch))

    model.train()
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device, dtype=torch.float32, non_blocking=True), label.to(device, dtype=torch.int64, non_blocking=True)
        if not args.pre_train:
            cpt, pred, att, update = model(data)
            retri_loss, quantity_loss = get_retrieval_loss(cpt, label, args.num_classes, device)
            loss_pred = F.nll_loss(F.log_softmax(pred, dim=1), label)
            acc = cal_acc(pred, label)
            batch_dis_loss = batch_cpt_discriminate(update, att)
            consistence_loss = att_consistence(update, att)
            att_binary_loss = att_binary(att)
            attn_loss = att_area_loss(att)
            att_dis_loss = att_discriminate(att)

            retri_losses.update(retri_loss.item())
            att_losses.update(attn_loss.item())
            q_losses.update(quantity_loss.item())
            batch_dis_losses.update(batch_dis_loss.item())
            consistence_losses.update(consistence_loss.item())
            att_binary_losses.update(att_binary_loss)
            att_dis_losses.update(att_dis_loss)
            pred_acces.update(acc)

            if epoch >= args.lr_drop:
                s = 0
                k = 2
                q = 5
                t = 2
            else:
                s = 0
                k = 1
                q = 1
                t = 0.5

            loss_total = retri_loss + s * attn_loss + t * quantity_loss + 0.5 * loss_pred + q * consistence_loss - k * batch_dis_loss + 0 * att_dis_loss
        else:
            cpt = model(data)
            retri_loss, quantity_loss = get_retrieval_loss(cpt, label, args.num_classes, device)
            retri_losses.update(retri_loss.item())
            q_losses.update(quantity_loss.item())
            loss_total = retri_loss + 0.1 * quantity_loss

            # retri_loss = F.nll_loss(F.log_softmax(cpt, dim=1), label)
            # acc = cal_acc(cpt, label)
            # q_losses.update(acc)
            # loss_total = retri_loss

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            progress.display(batch_idx)


def test_MAP(args, model, database_loader, test_loader, device):
    print('Waiting for generate the hash code from database')
    database_hash, database_labels, database_acc = predict_hash_code(args, model, database_loader, device)
    print('Waiting for generate the hash code from test set')
    test_hash, test_labels, test_acc = predict_hash_code(args, model, test_loader, device)
    print("label", database_labels.shape)
    print('Calculate MAP.....')

    MAP, R, APx = mean_average_precision(args, database_hash, test_hash, database_labels, test_labels)
    # print(MAP)
    # print(R)
    # print(APx)
    return MAP, test_acc