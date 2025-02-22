import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import SGD, Adam
from torchvision import datasets
import matplotlib.pyplot as plt
from torchsummary import summary

def get_model():
    model = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1, 1),
        nn.Sigmoid()
    )
    loss_func = nn.BCELoss()
    opt = Adam(model.parameters(), lr=0.001)
    return model, loss_func, opt

def get_data():
    x_np = np.random.randint(-8, 8, (2, 1, 4, 4))
    X_train = torch.tensor(x_np).float()
    Y_train = torch.tensor([0, 1]).float()
    return X_train, Y_train

def get_dl(X_train, Y_train):
    train_dl = DataLoader(TensorDataset(X_train, Y_train))
    return train_dl

def check_summary(model, x):
    summary(model, x)

def train_model(x, y, model, loss_func, opt):
    model.train()
    y_pred = model(x)
    loss_val = loss_func(y_pred.squeeze(0), y)
    loss_val.backward()
    opt.step()
    opt.zero_grad()
    return loss_val.item()

X_train, Y_train = get_data()
train_dl = get_dl(X_train, Y_train)
model, loss_func, opt = get_model()
# check_summary(model, X_train)

for _ in range(1000):
    for index, item in enumerate(iter(train_dl)):
        x, y = item
        response = train_model(x, y, model, loss_func, opt)

print('model(X_train[:1])', model(X_train[1]))

