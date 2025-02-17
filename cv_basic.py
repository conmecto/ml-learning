import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

def load_sample_img():
    img = cv2.imread('./data/flower_valley.jpg')
    img_shape = img.shape
    print('image', img_shape)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray_small = cv2.resize(img_gray, (50, 50))

    temp_image = img.copy()

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            # temp_image[i][j][0] = 0
            # temp_image[i][j][1] = 0
            temp_image[i][j][2] = 0

    plt.imshow(temp_image)
    plt.show()

#load_sample_img()

def load_img_dataset(download=False):
    data_path = './data/FMNIST'
    fmnist = datasets.FashionMNIST(data_path, download=download, train=True)
    tr_images = fmnist.data
    tr_targets = fmnist.targets
    print('Data size', len(tr_images))
    print('Shape', tr_images.shape)
    print('Classes', fmnist.classes)
    print('Unique values', tr_targets.unique())
    return tr_images, tr_targets

class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        float_x = x.float()
        reshaped_x = float_x.view(-1, 28 * 28)
        self.x, self.y = reshaped_x, y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

def get_data(x, y):
    fmnistDataset = FMNISTDataset(x, y)
    train_dl = DataLoader(fmnistDataset, batch_size=32, shuffle=True)
    return train_dl

def get_model():
    model = nn.Sequential(nn.Linear(28 * 28, 1000), nn.ReLU(), nn.Linear(1000, 10))
    loss_func = nn.CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=0.01)
    return model, loss_func, opt

def train_batch(x, y, model, opt, loss_func):
    model.train()
    y_pred = model(x)
    loss = loss_func(y_pred, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    y_pred = model(x)
    max_values, argmaxes = y_pred.max(-1)
    is_correct = argmaxes == y
    return is_correct.numpy().tolist()

tr_images, tr_targets = load_img_dataset(False)
train_dl = get_data(tr_images, tr_targets)
model, loss_func, opt = get_model()

losses, accuracies = [], []
for epoch in range(5):
    print('Epoch', epoch)
    epoch_losses, epoch_accuracies = [], []

    for index, batch in enumerate(iter(train_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, opt, loss_func)
        epoch_losses.append(batch_loss)
    
    for index, batch in enumerate(iter(train_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        epoch_accuracies.append(is_correct)
    
    epoch_loss = np.array(epoch_losses).mean()
    epoch_accuracy = np.mean(epoch_accuracies)
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)

epochs = np.arange(5)+1
plt.figure(figsize=(20,5))
plt.subplot(121)
plt.title('Loss value over increasing epochs')
plt.plot(epochs, losses, label='Training Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy value over increasing epochs')
plt.plot(epochs, accuracies, label='Training Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
plt.legend()
plt.show()



