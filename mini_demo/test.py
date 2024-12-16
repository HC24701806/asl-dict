import numpy as np
import torch
import csv
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def load_file_data(filename):
    res = []
    with open(f'mini_dataset/{filename}.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        
        num_lines = 0
        for line in reader:
            for i in range(1, 137):
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

test_data = Data(splits['train'], file_labels)
test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

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
model.load_state_dict(torch.load('model_v1.pth'))
dataiter = iter(test_loader)

correct, total = 0, 0
# no need to calculate gradients during inference
with torch.no_grad():
  for data in test_loader:
    inputs, labels = data
    # calculate output by running through the network
    outputs = model(inputs)
    # get the predictions
    __, predicted = torch.max(outputs.data, 1)
    # update results
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print(f'Accuracy of the network on the {len(test_data)} test data: {100 * correct // total} %')