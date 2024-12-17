import numpy as np
import torch
import csv
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def load_file_data(filename):
    res = np.zeros(13 * 136, dtype='float32')
    with open(f'mini_dataset/{filename}.txt') as input:
        data = input.readline().split('\t')
        index = 0
        for pt in data:
            if pt != '':
                res[index] = np.float32(pt)

    return torch.from_numpy(np.array(res))

class Data(Dataset):
    def __init__(self, data_files, labels):
        self.data_files = data_files
        self.labels = labels

    def __getitem__(self, index):
        filename = self.data_files[index]
        file = load_file_data(filename)
        label = self.labels[filename]
        return file, label
    
    def __len__(self):
        return len(self.data_files)

labels_dict = {}
with open('random_classes.txt', 'r') as txtfile:
    content = txtfile.readlines()
    i = 0
    for line in content:
        labels_dict[line[:-1]] = i
        i += 1

splits = {}
file_labels = {}
with open('augmented_sample.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)

    for line in reader:
        split = line[0]
        file = line[1]
        gloss = line[2]

        if split not in splits:
            splits[split] = []
        splits[split].append(file)

        file_labels[file] = labels_dict[gloss]

train_data = Data(splits['train'], file_labels)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

val_data = Data(splits['val'], file_labels)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13 * 136, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(300):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    val_correct = 0
    val_total = 0
    for data in val_loader:
        inputs, labels = data
        outputs = model(inputs)
        __, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
    print('Epoch: ' + str(epoch + 1) + '\tTraining loss: ' + str(round(running_loss/2000, 5)) + '\tValidation accuracy: ' + str(round(100 * val_correct/val_total, 1)) + '%')

torch.save(model.state_dict(), 'model_plus_v1.pth')

# testing
test_data = Data(splits['test'], file_labels)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

test_correct = 0
test_total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        __, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
print('Test accuracy: ' + str(round(100 * test_correct/test_total, 1)) + '%')