from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
import torch
from termcolor import colored
from utils.engine_recon import train, evaluation, vis_one
import torch.nn as nn
from configs import parser
from model.reconstruct.model_main import ConceptAutoencoder
import os


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    valset = datasets.MNIST('../data', train=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=False)
    valloader = DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=False)
    model = ConceptAutoencoder(args, num_concepts=args.num_cpt)
    reconstruction_loss = nn.MSELoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    device = torch.device("cuda:0")
    model.to(device)
    record_res = []
    record_att = []
    accs = []

    for i in range(args.epoch):
        print(colored('Epoch %d/%d' % (i + 1, args.epoch), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        if i == args.lr_drop:
            print("Adjusted learning rate to 0.00001")
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
        train(args, model, device, trainloader, reconstruction_loss, optimizer, i)
        res_loss, att_loss, acc = evaluation(model, device, valloader, reconstruction_loss)
        record_res.append(res_loss)
        record_att.append(att_loss)
        accs.append(acc)
        if i % args.fre == 0:
            vis_one(model, device, valloader, epoch=i, select_index=1)
        print("Reconstruction Loss: ", record_res)
        print("Acc: ", accs)
        torch.save(model.state_dict(), f"saved_model/mnist_model_cpt{args.num_cpt}.pt")


if __name__ == '__main__':
    args = parser.parse_args()
    args.att_bias = 5
    args.quantity_bias = 0.1
    args.distinctiveness_bias = 0
    args.consistence_bias = 0
    os.makedirs(args.output_dir + '/', exist_ok=True)
    main()
