# code based on https://medium.com/@enrico.randellini/hands-on-video-classification-with-pytorchvideo-dc9cfcc1eb5f

import numpy as np
import csv
import time
from tqdm import tqdm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import ClassificationDataset
from model_utils import train_batch, accuracy, val_loss

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.base_model.blocks[5].proj = nn.Sequential(nn.Linear(2048, 128),
                                                        nn.ReLU(),
                                                        nn.Dropout(0.3),
                                                        nn.Linear(128, 50))

    def forward(self, x):
        x = self.base_model(x)
        return x

t1 = time.time()

#load data
labels = {}
with open('sample_classes.txt') as labels_file:
    content = labels_file.readlines()
    i = 0
    for line in content:
        labels[line[:-1]] = i
        i += 1

splits = {}
label_list = np.empty(1525, dtype=int)
id_dict = {}
splits['train'] = np.empty(0, dtype=int)
splits['val'] = np.empty(0, dtype=int)
splits['test'] = np.empty(0, dtype=int)
with open('random_sample.csv') as data:
    reader = csv.reader(data)
    next(reader)

    id = 0
    for line in reader:
        split = line[0]
        file_name = line[1]
        label = line[2]

        splits[split] = np.append(splits[split], id)
        label_list[id] = labels[label]
        id_dict[id] = file_name
        id += 1

t2 = time.time()
print(f'loading data: {t2 - t1}s')

#prepare for training
train_dataset = ClassificationDataset(id_list=splits['train'], label_list=label_list, id_dict=id_dict)
val_dataset = ClassificationDataset(id_list=splits['val'], label_list=label_list, id_dict=id_dict)
test_dataset = ClassificationDataset(id_list=splits['test'], label_list=label_list, id_dict=id_dict)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

device = torch.device('mps')

model = Model().to('mps')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

train_epoch_losses, train_epoch_accuracies = [], []
val_epoch_losses, val_epoch_accuracies = [], []

t3 = time.time()
print(f'preparing training: {t3 - t2}s')

# train
for epoch in range(10):
    e_t1 = time.time()
    # iterate on all train batches of the current epoch by executing the train_batch function
    for inputs, labels in tqdm(train_dataloader, desc=f'epoch {str(epoch + 1)} | train'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_loss = train_batch(inputs, labels, model, optimizer, criterion)
        train_epoch_losses.append(batch_loss)
    train_epoch_loss = np.array(train_epoch_losses).mean()
    e_t2 = time.time()
    print(f'train (epoch {epoch}): {e_t2 - e_t1}s')

    # iterate on all train batches of the current epoch by calculating their accuracy
    for inputs, labels in tqdm(train_dataloader, desc=f'epoch {str(epoch + 1)} | train_acc'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        is_correct = accuracy(inputs, labels, model)
        train_epoch_accuracies.extend(is_correct)
    train_epoch_accuracy = np.mean(train_epoch_accuracies)
    e_t3 = time.time()
    print(f'get accuracy (epoch {epoch}): {e_t3 - e_t2}s')

    # iterate on all batches of val of the current epoch by calculating the accuracy and the loss function
    for inputs, labels in tqdm(val_dataloader, desc=f'epoch {str(epoch + 1)} | val'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        val_is_correct = accuracy(inputs, labels, model)
        val_epoch_accuracies.extend(val_is_correct)
        validation_loss = val_loss(inputs, labels, model, criterion)
        val_epoch_losses.append(validation_loss)
    val_epoch_accuracy = np.mean(val_epoch_accuracies)
    val_epoch_loss = np.mean(val_epoch_losses)
    e_t4 = time.time()
    print(f'val (epoch {epoch}): {e_t4 - e_t3}s')

    print(train_epoch_loss, train_epoch_accuracy)
    print(val_epoch_loss, val_epoch_accuracy)

    exp_lr_scheduler.step()
    torch.mps.empty_cache()
    print("---------------------------------------------------------")