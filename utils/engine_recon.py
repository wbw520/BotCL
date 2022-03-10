import torch
import torch.nn.functional as F
from utils.record import AverageMeter, ProgressMeter, show
from model.retrieval.loss import batch_cpt_discriminate, att_consistence, quantization_loss, att_area_loss


def cal_acc(preds, labels):
    with torch.no_grad():
        pred = preds.argmax(dim=-1)
        acc = torch.eq(pred, labels).sum().float().item() / labels.size(0)
        return acc


def train(model, device, loader, rec_loss, optimizer, epoch):
    recon_losses = AverageMeter('Reconstruction Loss', ':.4')
    att_losses = AverageMeter('Att Loss', ':.4')
    pred_losses = AverageMeter('Pred Loss', ':.4')
    batch_dis_losses = AverageMeter('Dis_loss_batch', ':.4')
    consistence_losses = AverageMeter('Consistence_loss', ':.4')
    q_losses = AverageMeter('Q_loss', ':.4')
    pred_acces = AverageMeter('Acc', ':.4')
    progress = ProgressMeter(len(loader),
                             [recon_losses, att_losses, pred_losses, pred_acces, batch_dis_losses, consistence_losses, q_losses],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        cpt, pred, out, att, update = model(data)

        loss_pred = F.nll_loss(F.log_softmax(pred, dim=1), label)
        acc = cal_acc(pred, label)
        reconstruction_loss = rec_loss(out.view(data.size(0), 1, 28, 28), data)
        quantity_loss = quantization_loss(cpt)
        batch_dis_loss = batch_cpt_discriminate(update, att)
        consistence_loss = att_consistence(update, att)
        att_loss = att_area_loss(att)

        recon_losses.update(reconstruction_loss.item())
        att_losses.update(att_loss.item())
        pred_losses.update(loss_pred.item())
        pred_acces.update(acc)
        q_losses.update(quantity_loss.item())
        batch_dis_losses.update(batch_dis_loss.item())
        consistence_losses.update(consistence_loss.item())

        loss_total = 1 * reconstruction_loss + 5 * att_loss + loss_pred + 0 * quantity_loss + 0 * batch_dis_loss + 0 * consistence_loss

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            progress.display(batch_idx)


@torch.no_grad()
def evaluation(model, device, loader, rec_loss):
    model.eval()
    record_res = 0.0
    record_att = 0.0
    accs = 0
    L = len(loader)

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        cpt, pred, out, att, update = model(data)

        acc = cal_acc(pred, label)
        reconstruction_loss = rec_loss(out.view(data.size(0), 28, 28), data)
        record_res += reconstruction_loss.item()
        att_loss = att_area_loss(att)
        record_att += att_loss.item()
        accs += acc
    return round(record_res/L, 4), round(record_att/L, 4), round(accs/L, 4)


def vis_one(model, device, loader, epoch=None, select_index=0):
    data, label = iter(loader).next()
    img_orl = data[select_index]
    img = img_orl.unsqueeze(0).to(device)
    pred = model(img)[2].view(28, 28).cpu().detach().numpy()
    show(img_orl.numpy()[0], pred, epoch)
