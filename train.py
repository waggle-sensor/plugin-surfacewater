import numpy as np
import argparse
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, action='store')
    parser.add_argument('--batch-size', type=int, default=2, action='store')
    parser.add_argument('--epochs', type=int, default=100, action='store')

    return parser.parse_args()


def train(args):
    stats_file = open("stats.txt", "a", buffering=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    transformation = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=args.data_dir,
                                   transform=transformation)

    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train, val = torch.utils.data.random_split(dataset, [n_train, n_val])
    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                               shuffle=True, num_workers=8, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val, batch_size=args.batch_size,
                                             shuffle=False, num_workers=8, pin_memory=True)

    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    steps = 0
    for epoch in range(args.epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(test_loss/len(valloader))


                stats = dict(
                      epoch=epoch+1,
                      train_loss=running_loss/print_every,
                      test_loss=test_loss/len(valloader),
                      test_accuracy=accuracy/len(valloader),
                      # output=logps.tolist(),
                      # label=labels.cpu().detach().numpy().tolist(),
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)



                running_loss = 0
                model.train()
        torch.save(model, 'model.pth')


if __name__ == '__main__':
    args = get_args()
    train(args)
