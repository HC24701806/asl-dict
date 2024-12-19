# code based on https://medium.com/@enrico.randellini/hands-on-video-classification-with-pytorchvideo-dc9cfcc1eb5f

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import csv
import cv2

import torch
import torch.nn as nn
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import DataLoader
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.base_model.blocks[5].proj = nn.Sequential(nn.Linear(2048, 128),
                                                        nn.ReLU(),
                                                        nn.Dropout(0.3),
                                                        nn.Linear(128, 49))

    def forward(self, x):
        x = self.base_model(x)
        return x
    
def get_video_data(input):
    video = cv2.VideoCapture('../butterfly.mp4')
    frame_list = sorted(np.int32(np.multiply(np.random.normal(0.4, 0.1667, 12), cv2.CAP_PROP_FRAME_COUNT)))
    res = np.zeros((12, 480, 640, 3), dtype='float32')
    i = 0
    for frame in frame_list:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = video.read()
        if not ret:
            break
        res[i] = np.float32(np.array(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 480))))
        i += 1
    video.release()
    res = np.transpose(res, [3, 0, 1, 2])
    return torch.from_numpy(res).shape

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, ids, label_list) -> None:
        super().__init__()
        self.ids = ids
        self.label_list = label_list

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        inputs = get_video_data(id)
        return inputs, label
    
post_act = torch.nn.Softmax(dim=1)

def train_batch(inputs, labels, model, optimizer, criterion):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

@torch.no_grad()
def accuracy(inputs, labels, model):
    model.eval()
    outputs = model(inputs)
    preds = post_act(outputs)
    _, pred_classes = torch.max(preds, 1)
    is_correct = pred_classes == labels
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def val_loss(inputs, labels, model, criterion):
    model.eval()
    outputs = model(inputs)
    val_loss = criterion(outputs, labels)
    return val_loss.item()

labels = {}
with open('sample_classes.txt') as labels_file:
    content = labels_file.readlines()
    i = 0
    for line in content:
        labels[line[:-1]] = i
        i += 1

splits = {}
label_list = {}
splits['train'] = []
splits['val'] = []
splits['test'] = []
with open('augmented_sample.csv') as data:
    reader = csv.reader(data)
    next(reader)

    for line in reader:
        split = line[0]
        id = line[1][:-4]
        label = line[2]

        splits[split].append(id)
        label_list[id] = labels[line[2]]

train_dataset = ClassificationDataset(ids=splits['train'], label_list=label_list)
val_dataset = ClassificationDataset(ids=splits['val'], label_list=label_list)
test_dataset = ClassificationDataset(ids=splits['test'], label_list=label_list)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

train_epoch_losses, train_epoch_accuracies = [], []
val_epoch_losses, val_epoch_accuracies = [], []

for epoch in range(10):
    # iterate on all train batches of the current epoch by executing the train_batch function
    for inputs, labels in tqdm(train_dataloader, desc=f'epoch {str(epoch + 1)} | train'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_loss = train_batch(inputs, labels, model, optimizer, criterion)
        train_epoch_losses.append(batch_loss)
    train_epoch_loss = np.array(train_epoch_losses).mean()

    # iterate on all train batches of the current epoch by calculating their accuracy
    for inputs, labels in tqdm(train_dataloader, desc=f'epoch {str(epoch + 1)} | train'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        is_correct = accuracy(inputs, labels, model)
        train_epoch_accuracies.extend(is_correct)
    train_epoch_accuracy = np.mean(train_epoch_accuracies)

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

    print(train_epoch_loss, train_epoch_accuracy)
    print(val_epoch_loss, val_epoch_accuracy)

    exp_lr_scheduler.step()
    torch.cuda.empty_cache()
    print("---------------------------------------------------------")