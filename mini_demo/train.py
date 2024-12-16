import numpy as np
import torch
import csv
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def load_file_data(filename):
    res = []
    print(filename)
    with open(f'dataset/{filename}.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        
        num_lines = 0
        for line in reader:
            for i in range(2, 138):
                res.append(np.float32(line[i]))
            num_lines += 1
        
        padding = 13 - num_lines
        for i in range(136 * padding):
            res.append(np.float32(0))

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
with open('stats.txt', 'r') as txtfile:
    content = txtfile.readlines()
    i = 0
    for line in content:
        if i == 100:
            break
        labels_dict[line.split(' ')[0]] = i
        i += 1

splits = {}
file_labels = {}
with open('random_sample.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)

    for line in reader:
        split = line[0]
        file = line[1][:-4]
        gloss = line[2]

        if split not in splits:
            splits[split] = []
        splits[split].append(file)

        file_labels[file] = labels_dict[gloss]

train_data = Data(splits['train'], file_labels)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

val_data = Data(splits['val'], file_labels)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True)

input_dim = 13 * 136
hidden_layers = 25
output_dim = 100

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, output_dim)
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x
    
model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(2):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Move data to the appropriate device (e.g., GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs, labels = inputs.to(device), labels.to(device)
        # set optimizer to zero grad to remove previous epoch gradients
        optimizer.zero_grad()
        # forward propagation
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # backward propagation
        loss.backward()
        # optimize
        optimizer.step()
        running_loss += loss.item()
    # display statistics
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/2000:.5f}')