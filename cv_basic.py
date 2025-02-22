import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import matplotlib.ticker as mtick
import matplotlib.ticker as mticker
import seaborn as sns


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
    print('Classes', fmnist.classes)
    print('Unique values', tr_targets.unique())
    print('Training data')
    print('Data size', len(tr_images))
    print('Shape', tr_images.shape)
    val_fmnist = datasets.FashionMNIST(data_path, download=download, train=False)
    val_images = val_fmnist.data
    val_targets = val_fmnist.targets
    print('Validation data')
    print('Data size', len(val_images))
    print('Shape', val_images.shape)
    return fmnist, tr_images, tr_targets, val_images, val_targets

class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        float_x = x.float() / 255
        reshaped_x = float_x.view(-1, 28 * 28)
        self.x, self.y = reshaped_x, y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

def get_data(tr_images, tr_targets, val_images, val_targets):
    train = FMNISTDataset(tr_images, tr_targets)
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    val = FMNISTDataset(val_images, val_targets)
    val_dl = DataLoader(val, batch_size=len(val_images), shuffle=False)
    return train_dl, val_dl

def get_model():
    model = nn.Sequential(
        # nn.Dropout(0.25),
        nn.Linear(28 * 28, 1000), 
        nn.ReLU(), 
        nn.BatchNorm1d(1000),
        # nn.Dropout(0.25),
        nn.Linear(1000, 10)
        )
    loss_func = nn.CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=0.001)
    return model, loss_func, opt

def train_batch(x, y, model, opt, loss_func):
    model.train()
    y_pred = model(x)
    # l1_regularization = 0
    # for param in model.parameters():
    #     l1_regularization += torch.norm(param, 1)
    # loss = loss_func(y_pred, y) + 0.0001 * l1_regularization
    l2_regularization = 0
    for param in model.parameters():
        l2_regularization += torch.norm(param, 2)
    loss = loss_func(y_pred, y) + 0.01 * l2_regularization
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

@torch.no_grad()
def cal_val_loss(x, y, model, loss_func):
    model.eval()
    y_pred = model(x)
    val_loss = loss_func(y_pred, y)
    return val_loss.item()

@torch.no_grad()
def check_for_rolled_images(img, model, fmnist):
    img2 = (img / 255)
    img3 = img2.view(28, 28)
    preds = []
    for px in range(-5, 6):
        img4 = np.roll(img3, px, axis=1)
        img5 = torch.tensor(img4).view(28 * 28)
        temp = model(img5)
        print('model(img5)', temp)
        detached_temp = temp.detach()
        print('detached_temp', detached_temp)
        pred = detached_temp.numpy()
        print('pred', pred)
        # pred = model(img5).detach().numpy()
        preds.append(np.exp(pred)/np.sum(np.exp(pred)))
    fig, ax = plt.subplots(1,1, figsize=(12,10))
    plt.title('Probability of each class for various translations')
    sns.heatmap(np.array(preds), annot=True, ax=ax, fmt='.2f', xticklabels=fmnist.classes,yticklabels=[str(i)+ str(' pixels') for i in range(-5,6)], cmap='gray')


fmnist, tr_images, tr_targets, val_images, val_targets = load_img_dataset(False)
train_dl, val_dl = get_data(tr_images, tr_targets, val_images, val_targets)
model, loss_func, opt = get_model()

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
for epoch in range(10):
    print('Epoch', epoch)
    train_epoch_losses, train_epoch_accuracies = [], []

    for index, batch in enumerate(iter(train_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, opt, loss_func)
        train_epoch_losses.append(batch_loss)
    
    for index, batch in enumerate(iter(train_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        train_epoch_accuracies.append(is_correct)

    for index, batch in enumerate(iter(val_dl)):
        x, y = batch
        val_is_correct = accuracy(x, y, model)
        val_loss = cal_val_loss(x, y, model, loss_func)

    epoch_loss = np.array(train_epoch_losses).mean()
    train_losses.append(epoch_loss)
    epoch_accuracy = np.mean(train_epoch_accuracies)
    train_accuracies.append(epoch_accuracy)
    val_losses.append(val_loss)
    val_epoch_accuracy = np.mean(val_is_correct)
    val_accuracies.append(val_epoch_accuracy)

check_for_rolled_images(tr_images[26783], model, fmnist)


# epochs = np.arange(10)+1

# plt.subplot(211)
# plt.plot(epochs, train_losses, 'bo', label='Training loss')
# plt.plot(epochs, val_losses, 'r', label='Validation loss')
# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
# plt.title('Training and validation loss with Adam optimizer')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid('off')
# plt.show()
# plt.subplot(212)
# plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
# plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
# plt.title('Training and validation accuracy with Adam optimizer')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
# plt.legend()
# plt.grid('off')
# plt.show()



