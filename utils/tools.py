import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import copy
import matplotlib.cm as mpl_color_map
import cv2


def cal_acc(preds, labels, p):
    with torch.no_grad():
        pred = preds.argmax(dim=1)
        if p:
            print(pred)
            print(labels)
            print("---------------")
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
            if args.dataset != "matplot":
                acc = cal_acc(pred, label, False)
            else:
                pred = F.sigmoid(pred)
                acc = torch.eq(pred.round(), label).sum().float().item() / pred.shape[0] / pred.shape[1]
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
    MAP, R, APx = mean_average_precision(args, database_hash, test_hash, database_labels, test_labels)

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
    sim = np.matmul(database_hash, test_hash.T)
    ids = np.argsort(-sim, axis=0)
    idx = ids[:, 0]
    ids = idx[:R]
    return ids


def attention_estimation(data, label, model, transform, device, name):
    selected_class = name
    contains = []
    for i in range(len(data)):
        if selected_class in data[i]:
            print(i)
            contains.append(data[i])

    attention_record = []
    for i in range(len(contains)):
        print(contains[i])
        img_orl = Image.open(contains[i]).convert('RGB')
        cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, None)
        attention_record.append((torch.tanh(att.sum(-1))).squeeze(0).cpu().detach().numpy())
    return np.array(attention_record)


def attention_estimation_mnist(data, target, model, transform, transform2, device, name):
    selected_class = name
    contains = []
    for i in range(len(target)):
        if selected_class == int(target[i]):
            contains.append(data[i])

    attention_record = []
    for i in range(len(contains)):
        img_orl = data[i]
        img_orl = Image.fromarray(img_orl.numpy())
        pred, x, att_loss, pp = model(transform2(transform(img_orl)).unsqueeze(0).to(device), None, None)
        attention_record.append((torch.tanh(pp)).squeeze(0).cpu().detach().numpy())
    return np.array(attention_record)


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def make_grad(args, extractor, output, img_heat, grad_min_level, save_name, target_index, segment=None):
    # img_heat = img_heat.resize((args.img_size, args.img_size), Image.BILINEAR)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    # target_index = None
    mask = extractor(target_index, output).cpu().unsqueeze(0).unsqueeze(0)
    mask = F.interpolate(mask, size=(args.img_size, args.img_size), mode="bilinear")
    mask = mask.squeeze(dim=0).squeeze(dim=0)
    mask = mask.detach().numpy()
    mask = np.maximum(mask, 0)
    mask = mask - np.min(mask)
    mask = mask / np.max(mask)
    mask = np.maximum(mask, grad_min_level)
    mask = mask - np.min(mask)
    mask = mask / np.max(mask)
    if segment is not None:
        mask = mask * segment
    # show_cam_on_image(img_heat, mask, target_index, save_name)
    return mask


def show_cam_on_image(img, masks, target_index, save_name):
    final = np.uint8(255*masks)

    mask_image = Image.fromarray(final, mode='L')
    mask_image.save(f'vis_compare/{save_name}_{target_index}_mask.png')

    heatmap_only, heatmap_on_image = apply_colormap_on_image(img, final, 'jet')
    heatmap_on_image.save(f'vis_compare/{save_name}_{target_index}.png')


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def shot_game(mask, segment_name):
    mask = np.array(mask)
    names = segment_name.split("/")
    new_name = "/media/wbw/a7f02863-b441-49d0-b546-6ef6fefbbc7e/CUB200/segmentations/" + names[-2] + "/" + names[-1][:-4] + ".png"
    segment = np.array(cv2.imread(new_name, cv2.IMREAD_UNCHANGED))
    if segment.shape[-1] == 4:
        print("--------------------------")
        return None, None
    segment = cv2.resize(segment, (224, 224), interpolation=cv2.INTER_NEAREST)
    segment[segment > 0] = 1
    overlap_seg = segment * mask
    hitted = np.sum(overlap_seg) / np.sum(mask)
    return hitted, segment


