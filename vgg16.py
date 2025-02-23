import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from torchvision import transforms,models,datasets
# !pip install torch_summary
import matplotlib.pyplot as plt
from torchsummary import summary
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import kagglehub

path = kagglehub.dataset_download("tongpython/cat-and-dog")

train_data_dir = '/root/.cache/kagglehub/datasets/tongpython/cat-and-dog/versions/1/training_set/training_set'
test_data_dir = '/root/.cache/kagglehub/datasets/tongpython/cat-and-dog/versions/1/test_set/test_set'

model = models.vgg16(pretrained=True).to(device)

summary(model, torch.zeros(1, 3, 224, 224))

model

from PIL import Image
from torch import optim
import cv2, glob, numpy as np, pandas as pd
from glob import glob
import torchvision.transforms as transforms

class CatsDogsDataset(Dataset):
  def __init__(self, folder):
    cats = glob(folder + '/cats/*.jpg')
    dogs = glob(folder + '/dogs/*.jpg')
    self.fpaths = cats[:500] + dogs[:500]
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224,  0.225]) # apply z-score normalization
    shuffle(self.fpaths)
    self.targets=[fpath.split('/')[-1].startswith('dog') for fpath in self.fpaths]

  def __len__(self):
    return len(self.fpaths)

  def __getitem__(self, ix):
    f = self.fpaths[ix]
    target = self.targets[ix]
    im = (cv2.imread(f)[:,:,::-1])
    im = cv2.resize(im, (224,224))
    im = torch.tensor(im/255)
    im = im.permute(2, 0, 1)
    im = self.normalize(im)
    return im.to(device).float(), torch.tensor([target]).float().to(device)

data = CatsDogsDataset(train_data_dir)
im, label = data[200]
plt.imshow(im.permute(1, 2, 0).cpu())
print(label)

def get_model():
  model = models.vgg16(pretrained=True)
  for param in model.parameters():
    param.requires_grad = False
  model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
  model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
    nn.Sigmoid())
  loss_fn = nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(),lr= 1e-3)
  return model.to(device), loss_fn, optimizer

model, criterion, optimizer = get_model()
summary(model, torch.zeros(1,3,224,224))

def train_batch(x, y, model, opt, loss_fn):
  model.train()
  prediction = model(x)
  batch_loss = loss_fn(prediction, y)
  batch_loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
  model.eval()
  prediction = model(x)
  is_correct = (prediction > 0.5) == y
  return is_correct.cpu().numpy().tolist()

def get_data():
  train = CatsDogsDataset(train_data_dir)
  trn_dl = DataLoader(train, batch_size=32, shuffle=True, drop_last = True)
  val = CatsDogsDataset(test_data_dir)
  val_dl = DataLoader(val, batch_size=32, shuffle=True, drop_last = True)
  return trn_dl, val_dl

trn_dl, val_dl = get_data()
model, loss_fn, optimizer = get_model()

train_losses, train_accuracies = [], []
val_accuracies = []
for epoch in range(5):
  print(f" epoch {epoch + 1}/5")
  train_epoch_losses, train_epoch_accuracies = [], []
  val_epoch_accuracies = []
  for ix, batch in enumerate(iter(trn_dl)):
    x, y = batch
    batch_loss = train_batch(x, y, model, optimizer, loss_fn)
    train_epoch_losses.append(batch_loss)
  train_epoch_loss = np.array(train_epoch_losses).mean()
  for ix, batch in enumerate(iter(trn_dl)):
    x, y = batch
    is_correct = accuracy(x, y, model)
    train_epoch_accuracies.extend(is_correct)
  train_epoch_accuracy = np.mean(train_epoch_accuracies)
  for ix, batch in enumerate(iter(val_dl)):
    x, y = batch
    val_is_correct = accuracy(x, y, model)
    val_epoch_accuracies.extend(val_is_correct)
  val_epoch_accuracy = np.mean(val_epoch_accuracies)
  train_losses.append(train_epoch_loss)
  train_accuracies.append(train_epoch_accuracy)
  val_accuracies.append(val_epoch_accuracy)

# Commented out IPython magic to ensure Python compatibility.
epochs = np.arange(5)+1
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# %matplotlib inline
plt.plot(epochs, train_accuracies, 'bo',
         label='Training accuracy')
plt.plot(epochs, val_accuracies, 'r',
         label='Validation accuracy')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation accuracy \
with VGG16 \nand 1K training data points')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.95,1)
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) \
                           for x in plt.gca().get_yticks()])
plt.legend()
plt.grid('off')
plt.show()