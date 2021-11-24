from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
import torch
from termcolor import colored
from utils.engine_recon import train, evaluation, vis_one
import torch.nn as nn
from configs import parser
from model.reconstruct.model_main import ConceptAutoencoder


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    valset = datasets.MNIST('../data', train=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=False)
    valloader = DataLoader(valset, batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=False)
    model = ConceptAutoencoder(args, num_concepts=args.num_cpt)
    reconstruction_loss = nn.MSELoss()
    # reconstruction_loss = nn.L1Loss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    device = torch.device("cuda:0")
    model.to(device)
    record_res = []
    record_att = []
    accs = []

    for i in range(epoch):
        print(colored('Epoch %d/%d' % (i + 1, epoch), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        if i == args.lr_drop:
            print("Adjusted learning rate to 0.00001")
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
        train(model, device, trainloader, reconstruction_loss, optimizer, i)
        res_loss, att_loss, acc = evaluation(model, device, valloader, reconstruction_loss)
        record_res.append(res_loss)
        record_att.append(att_loss)
        accs.append(acc)
        if i % args.fre == 0:
            vis_one(model, device, valloader, epoch=i, select_index=1)
        print(record_res)
        print(record_att)
        print(accs)
        torch.save(model.state_dict(), f"saved_model/mnist_model_cpt{args.num_cpt}.pt")


if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = args.batch_size
    num_workers = args.num_workers
    epoch = args.epoch
    main()
