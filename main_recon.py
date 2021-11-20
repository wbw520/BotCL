from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
import torch
from termcolor import colored
from engine import train, evaluation, vis_one
import torch.nn as nn
from utils import get_optimizer, adjust_learning_rate
from model import ConceptAutoencoder


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
    model = ConceptAutoencoder(num_concepts=10)
    reconstruction_loss = nn.MSELoss()
    # reconstruction_loss = nn.L1Loss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.001)
    device = torch.device("cuda:0")
    model.to(device)
    record_res = []
    record_att = []
    accs = []

    for i in range(epoch):
        print(colored('Epoch %d/%d' % (i + 1, epoch), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        if i == 20:
            print("Adjusted learning rate to 0.00001")
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
        train(model, device, trainloader, reconstruction_loss, optimizer, i)
        res_loss, att_loss, acc = evaluation(model, device, valloader, reconstruction_loss)
        record_res.append(res_loss)
        record_att.append(att_loss)
        accs.append(acc)
        vis_one(model, device, valloader, epoch=i, select_index=1)
        print(record_res)
        print(record_att)
        print(accs)
        torch.save(model.state_dict(), "saved_models/mnist_model.pt")


if __name__ == '__main__':
    batch_size = 256
    num_workers = 4
    epoch = 50
    main()
