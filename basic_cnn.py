import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import SGD, Adam
from torchvision import datasets
import matplotlib.pyplot as plt
from torchsummary import summary
import matplotlib.ticker as mticker
import seaborn as sns

def get_model():
    model = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3200, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, opt

def get_data():
    path = './data/FMNIST'
    train_fmnist = datasets.FashionMNIST(path, download=False, train=True)
    x_train = train_fmnist.data
    y_train = train_fmnist.targets
    val_fmnist = datasets.FashionMNIST(path, download=False, train=False)
    x_val = val_fmnist.data
    y_val = val_fmnist.targets
    return train_fmnist, x_train, y_train, x_val, y_val

class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        float_x = x.float() / 255
        reshaped_x = float_x.view(-1, 1, 28, 28)
        self.x, self.y = reshaped_x, y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

def get_dl(x_train, y_train, x_val, y_val):
    train = FMNISTDataset(x_train, y_train)
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    val = FMNISTDataset(x_val, y_val)
    val_dl = DataLoader(val, batch_size=len(x_val), shuffle=False)
    return train_dl, val_dl

def check_summary(model, x):
    summary(model, x)

def train_batch(x, y, model, loss_fn, opt):
    model.train()
    y_pred = model(x)
    loss_val = loss_fn(y_pred, y)
    loss_val.backward()
    opt.step()
    opt.zero_grad()
    return loss_val.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    y_pred = model(x)
    max_values, argmaxes = y_pred.max(-1)
    is_correct = argmaxes == y
    return is_correct.numpy().tolist()

@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    return loss.item()

train_fmnist, x_train, y_train, x_val, y_val = get_data()
train_dl, val_dl = get_dl(x_train, y_train, x_val, y_val)
model, loss_fn, opt = get_model()
# check_summary(model, torch.zeros(1, 1, 28, 28))

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
for epoch in range(5):
    print('epoch', epoch)
    train_epoch_losses, train_epoch_accuracies = [], []
    for ix, batch in enumerate(iter(train_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, loss_fn, opt)
        train_epoch_losses.append(batch_loss)        
    train_epoch_loss = np.array(train_epoch_losses).mean()

    # for ix, batch in enumerate(iter(train_dl)):
    #     x, y = batch
    #     is_correct = accuracy(x, y, model)
    #     train_epoch_accuracies.extend(is_correct)
    # train_epoch_accuracy = np.mean(train_epoch_accuracies)

    # for ix, batch in enumerate(iter(val_dl)):
    #     x, y = batch
    #     val_is_correct = accuracy(x, y, model)
    #     validation_loss = val_loss(x, y, model, loss_fn)
    # val_epoch_accuracy = np.mean(val_is_correct)

    train_losses.append(train_epoch_loss)
    # train_accuracies.append(train_epoch_accuracy)
    # val_losses.append(validation_loss)
    # val_accuracies.append(val_epoch_accuracy)


# epochs = np.arange(5)+1
# plt.subplot(211)
# plt.plot(epochs, train_losses, 'bo', label='Training loss')
# plt.plot(epochs, val_losses, 'r', label='Validation loss')
# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
# plt.title('Training and validation loss with CNN')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid('off')
# plt.show()
# plt.subplot(212)
# plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
# plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
# plt.title('Training and validation accuracy with CNN')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# #plt.ylim(0.8,1)
# plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
# plt.legend()
# plt.grid('off')
# plt.show()

preds = []
ix = 24300
for px in range(-5,6):
    img = x_train[ix]/255.
    img = img.view(28, 28)
    img2 = np.roll(img, px, axis=1)
    plt.imshow(img2)
    plt.show()
    img3 = torch.Tensor(img2).view(-1,1,28,28)
    np_output = model(img3).detach().numpy()
    preds.append(np.exp(np_output)/np.sum(np.exp(np_output)))

fig, ax = plt.subplots(1,1, figsize=(12,10))
plt.title('Probability of each class for various translations')
sns.heatmap(np.array(preds).reshape(11,10), annot=True, ax=ax,fmt='.2f', xticklabels=train_fmnist.classes, yticklabels=[str(i)+str(' pixels') for i in range(-5,6)], cmap='gray')
