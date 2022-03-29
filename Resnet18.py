# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


if __name__ == '__main__':

    plt.ion()

    writer = SummaryWriter('runs/original')

    # Data transformation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    LOAD_PATH = './user_collection/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(LOAD_PATH, x),
                                              data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    class_len = len(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # wait for renewal

    def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            # Each epoch has train and test phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate data
                for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase])):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Set parameter grad to 0
                    optimizer.zero_grad()

                    # Feed-forward
                    # Trace calculate when train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Back-propagation and optimalization when train
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

    #                 ## Check each step
    #                 print(f'inputs : {inputs}')
    #                 print(f'labels : {labels}')
    #                 print(f'outputs : {outputs}')
    #                 print(f'outputs_size : {outputs.shape}')
    #                 print(f'torch.max(outputs, 1) : {torch.max(outputs, 1)}')
    #                 print(f'preds : {preds}')
    #                 print(f'loss : {loss}')
    #                 print(f'loss.item() : {loss.item()}')
    #                 print()
    #                 print(f'inputs.size(0) : {inputs.size(0)}')
    #                 print(f'running_loss : {running_loss}')
    #                 print(f'running_corrects : {running_corrects}')
    #                 raise Exception()

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                writer.add_scalar(f'{phase} loss', epoch_loss, epoch+1)
                writer.add_scalar(f'{phase} acc', epoch_acc, epoch+1)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # Deep copy best model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))

        # Load best model
        model.load_state_dict(best_model_wts)
        return model

    # Overall re-learning on the learned network. Slow but high accuracy.
    # The model that only trains the last layer is not very accurate.
    model_ft = models.resnet18(pretrained=True)

    # Size of last FC layer is len(class)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, class_len)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe all parameters are optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Every 7 epoch, reduce lr
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           exp_lr_scheduler, num_epochs=25)
