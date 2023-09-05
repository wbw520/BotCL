from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from model.reconstruct.model_main import ConceptAutoencoder
import torch
import os
from configs import parser
from PIL import Image
from utils.draw_tools import draw_bar, draw_plot
import numpy as np
import shutil
from utils.tools import attention_estimation_mnist
from utils.record import apply_colormap_on_image, show

shutil.rmtree('vis/', ignore_errors=True)
shutil.rmtree('vis_pp/', ignore_errors=True)
os.makedirs('vis/', exist_ok=True)
os.makedirs('vis_pp/', exist_ok=True)


def main():
    transform = transforms.Compose([transforms.ToTensor()])
    transform2 = transforms.Normalize((0.1307,), (0.3081,))
    val_imgs = datasets.MNIST('./data', train=False, download=True, transform=None).data
    val_target = datasets.MNIST('./data', train=False, download=True, transform=None).targets
    valset = datasets.MNIST('../data', train=False, transform=transform)
    valloader = DataLoader(valset, batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=False)
    model = ConceptAutoencoder(args, num_concepts=args.num_cpt, vis=True)
    device = torch.device("cuda:0")
    model.to(device)
    checkpoint = torch.load(f"saved_model/mnist_model_cpt{args.num_cpt}.pt", map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    data, label = next(iter(valloader))

    index = args.index

    img_orl = Image.fromarray((data[index][0].cpu().detach().numpy()*255).astype(np.uint8), mode='L')
    img_orl.save("vis/" + f'origin.png')
    img = data[index].unsqueeze(0).to(device)
    cpt, pred, cons, att_loss, pp = model(transform2(img))
    # print(torch.softmax(pred, dim=-1))
    print("The prediction is: ", torch.argmax(pred, dim=-1))
    cons = cons.view(28, 28).cpu().detach().numpy()
    show(data[index].numpy()[0], cons)

    for id in range(args.num_cpt):
        slot_image = np.array(Image.open(f'vis/0_slot_{id}.png'), dtype=np.uint8)
        heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl, slot_image, 'jet')
        heatmap_on_image.save("vis/" + f'0_slot_mask_{id}.png')

    # att_record = attention_estimation_mnist(val_imgs, val_target, model, transform, transform2, device, name=7)
    # print(att_record.shape)
    # draw_plot(att_record, "7")

    if args.deactivate == -1:
        is_start = True
        for batch_idx, (data, label) in enumerate(valloader):
            data, label = transform2(data).to(device), label.to(device)
            cpt, pred, out, att_loss, pp = model(data, None, "pass")

            if is_start:
                all_output = cpt.cpu().detach().float()
                all_label = label.unsqueeze(-1).cpu().detach().float()
                is_start = False
            else:
                all_output = torch.cat((all_output, cpt.cpu().detach().float()), 0)
                all_label = torch.cat((all_label, label.unsqueeze(-1).cpu().detach().float()), 0)

        all_output = all_output.numpy().astype("float32")
        all_label = all_label.squeeze(-1).numpy().astype("float32")

        print("cpt visualization")
        for j in range(args.num_cpt):
            root = 'vis_pp/' + "cpt" + str(j+1) + "/"
            os.makedirs(root, exist_ok=True)
            selected = all_output[:, j]
            ids = np.argsort(-selected, axis=0)
            idx = ids[:args.top_samples]
            for i in range(len(idx)):
                img_orl = val_imgs[idx[i]]
                img_orl = Image.fromarray(img_orl.numpy())
                img_orl.save(root + f'/origin_{i}.png')
                img = transform2(transform(img_orl))
                cpt, pred, out, att_loss, pp = model(img.unsqueeze(0).to(device), ["vis", root], [j, i])
                slot_image = np.array(Image.open(root + f'{i}.png'), dtype=np.uint8)
                heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl, slot_image, 'jet')
                heatmap_on_image.save(root + f'mask_{i}.png')


if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = 1000
    num_workers = args.num_workers
    epoch = args.epoch
    main()