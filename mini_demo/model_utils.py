import torch

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