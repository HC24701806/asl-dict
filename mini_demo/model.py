# code based on https://medium.com/@enrico.randellini/hands-on-video-classification-with-pytorchvideo-dc9cfcc1eb5f
# remember to set "export PYTORCH_ENABLE_MPS_FALLBACK=1" before running

import numpy as np
import csv
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import ClassificationDataset

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

#load data
labels = {}
with open('sample_classes.txt') as labels_file:
    content = labels_file.readlines()
    i = 0
    for line in content:
        labels[line[:-1]] = i
        i += 1

splits = {}
label_list = np.empty(0, dtype=int)
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
        label_list = np.append(label_list, labels[label])
        id_dict[id] = file_name
        id += 1

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

# train
for epoch in range(10):
    # iterate on all train batches of the current epoch by executing the train_batch function
    for inputs, labels in tqdm(train_dataloader, desc=f'epoch {str(epoch + 1)} | train'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_loss = train_batch(inputs, labels, model, optimizer, criterion)
        train_epoch_losses.append(batch_loss)
    train_epoch_loss = np.array(train_epoch_losses).mean()

    # iterate on all train batches of the current epoch by calculating their accuracy
    for inputs, labels in tqdm(train_dataloader, desc=f'epoch {str(epoch + 1)} | train_acc'):
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

    save_obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': exp_lr_scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(save_obj, f'./models/v1_{epoch + 1}.pth')
    exp_lr_scheduler.step()
    torch.mps.empty_cache()
    print('---------------------------------------------------------')

#test
actual = []
predicted = []
total = 0
model = model.eval()
with torch.no_grad():
    # cycle on all train batches of the current epoch by calculating their accuracy
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        # Get the predicted classes
        preds = post_act(outputs)
        _, pred_classes = torch.max(preds, 1)
        actual.extend(labels.cpu().numpy().tolist())
        predicted.extend(pred_classes.cpu().numpy().tolist())
        numero_video = len(labels.cpu().numpy().tolist())
        total += numero_video

    # report predictions and true values to numpy array
    print('Number of tested videos: ', total)
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    print('Accuracy: ', accuracy_score(actual, predicted))
    print(metrics.classification_report(actual, predicted))

    ## Plot confusion matrix
    cm = metrics.confusion_matrix(actual, predicted)

    fig, ax = plt.subplots(figsize=(50, 30))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=labels.keys(),
            yticklabels=labels.keys(), title="Confusion matrix")
    plt.yticks(rotation=0)
    fig.savefig('./models/confusion_matrix.png')

    ## Save report in a txt
    target_names = list(labels.keys())
    cr = metrics.classification_report(actual, predicted, target_names=target_names)
    with open('./models/report.txt', 'w') as report:
        report.write('Title\n\nClassification Report\n\n{}'.format(cr))
    report.close()