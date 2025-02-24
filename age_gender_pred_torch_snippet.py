# -*- coding: utf-8 -*-
"""age_gender_pred_torch_snippet.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WE4DZ-N2xs1MVy1uqvr-7LJjlGlPw2w2
"""

!pip install torch_summary
!pip install torchvision

!pip install torch_snippets

import torch
from torch import nn
from torchvision import models, datasets, transforms
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import glob, os, numpy as np, pandas as pd, cv2
from copy import deepcopy
from torch_snippets import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import kagglehub

path = kagglehub.dataset_download("mehmoodsheikh/fairface-dataset")
path = path + '/FairFace'

train_df = pd.read_csv(path + '/fairface_label_train.csv')
test_df = pd.read_csv(path + '/fairface_label_val.csv')

train_df

class GenderAgeClass(Dataset):
  def __init__(self, df):
    self.df = df
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    image_path = path + self.df.iloc[index, 0]
    # 1 -> read file as color
    img = read(image_path, 1)
    age = self.df.iloc[index, 1]
    gender = self.df.iloc[index, 2] == 'Female'
    return img, age, gender

  def preprocess_img(self, img):
    img = resize(img, (224, 224))
    img = torch.tensor(img).permute(2, 0, 1)
    img = self.normalize(img/255.)
    return img.unsqueeze(0)

  def collate_fn(self, batch):
    imgs, ages, genders = [], [], []
    for img, age, gender in batch:
      img = self.preprocess_img(img)
      genders.append(float(gender))
      ages.append(float(int(age)/80))
    ages, genders = [torch.tensor(x).to(device).float() for x in [ages, genders]]
    return imgs, ages, genders

train_ds = GenderAgeClass(train_df)
test_ds = GenderAgeClass(test_df)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=train_ds.collate_fn)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=True, collate_fn=test_ds.collate_fn)
im, gen, age = train_ds[0]
show(im, title=f'Gender: {gen}\nAge: {age}', sz=5)
inspect(ims, gens, ages)

ims, gens, ages = next(iter(train_loader))
inspect(ims, gens, ages)

class AgeGenderClassifier(nn.Module):
  def __init__(self):
    super(AgeGenderClassifier, self).__init__()
    self.intermediate = nn.Sequential(
        nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 128),
        nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 64), nn.ReLU()
        )
    self.age_classifier = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    self.gender_classifier = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

  def forward(self, x):
    x = self.intermediate(x)
    age = self.age_classifier(x)
    gender = self.gender_classifier(x)
    return gender, age

def get_model():
  model = models.vgg16(pretrained=True)
  for params in model.parameters():
    params.requires_grad = False
  model.avgpool = nn.Sequential(nn.Conv2d(512, 512, 3), nn.MaxPool2d(2), nn.ReLU(), nn.Flatten())
  model.classifier = AgeGenderClassifier()
  gender_criterion = nn.BCELoss()
  age_criterion = nn.L1Loss()
  loss_functions = gender_criterion, age_criterion
  optimizer = torch.optim.Adam(model.parameters(),lr= 1e-4)
  return model.to(device), loss_functions, optimizer

def train_batch(data, model, optimizer, criterion):
  model.train()
  optimizer.zero_grad()
  imgs, ages, genders = data
  genders_pred, ages_pred = model(imgs)
  gender_criterion, age_criterion = criterion
  gender_loss = gender_criterion(genders_pred.squeeze(), genders)
  age_loss = age_criterion(ages_pred.squeeze(), ages)
  total_loss = gender_loss + age_loss
  total_loss.backward()
  optimizer.step()
  return total_loss

def validate_batch(data, model, criterion):
  model.eval()
  imgs, ages, genders = data
  with torch.no_grad():
    pred_gender, pred_age = model(imgs)
  gender_criterion, age_criterion = criterion
  gender_loss = gender_criterion(pred_gender.squeeze(), genders)
  age_loss = age_criterion(pred_age.squeeze(), ages)
  total_loss = gender_loss + age_loss
  pred_gender = (pred_gender > 0.5).squeeze()
  gender_acc = (pred_gender == genders).float().sum()
  age_mae = torch.abs(ages - pred_age).float().sum()
  return total_loss, gender_acc, age_mae

model, criterion, optimizer = get_model()

model, criterion, optimizer = get_model()
n_epochs = 5
log = Report(n_epochs)
for epoch in range(n_epochs):
  N = len(train_dl)
  for ix, data in enumerate(train_dl):
    total_loss,gender_loss,age_loss = train_batch(data,
                              model, optimizer, criterion)
    log.record(epoch+(ix+1)/N, trn_loss=total_loss, end='\r')
  N = len(test_dl)
  for ix, data in enumerate(test_dl):
    total_loss,gender_acc,age_mae = validate_batch(data, model, criterion)
    gender_acc /= len(data[0])
    age_mae /= len(data[0])
    log.record(epoch+(ix+1)/N, val_loss=total_loss,
                val_gender_acc=gender_acc,
                val_age_mae=age_mae, end='\r')
  log.report_avgs(epoch+1)
log.plot_epochs()