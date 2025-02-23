# !pip install torch_summary
# !pip install torchvision

import torch
from torch import nn
from torchvision import models, datasets, transforms
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import glob, os, numpy as np, pandas as pd, cv2
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# !git clone https://github.com/udacity/P1_Facial_Keypoints.git
# !cd 'P1_Facial_Keypoints'

root_dir = 'P1_Facial_Keypoints/data/training/'
image_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
data = pd.read_csv('P1_Facial_Keypoints/data/training_frames_keypoints.csv')

class FacialKeypointsDataset(Dataset):
  def __init__(self, df):
    self.df = df
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    image_path = root_dir + self.df.iloc[index, 0]
    img = cv2.imread(image_path)/255.
    img = self.preprocess_img(img)
    kp = deepcopy(self.df.iloc[index, 1:].tolist())
    kp_x = (np.array(kp[0::2])/img.shape[1]).tolist()
    kp_y = (np.array(kp[1::2])/img.shape[0]).tolist()
    kp2 = kp_x + kp_y
    kp2 = torch.tensor(kp2)
    return img, kp2

  def preprocess_img(self, img):
    img = cv2.resize(img, (224, 224))
    img = torch.tensor(img).permute(2, 0, 1)
    img = self.normalize(img).float()
    return img.to(device)

  def load_img(self, ix):
    img_path = 'P1_Facial_Keypoints/data/training/' + self.df.iloc[ix,0]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
    img = cv2.resize(img, (224,224))
    return img

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(data, test_size=0.2, random_state=101)
train_ds = FacialKeypointsDataset(train_df.reset_index(drop=True))
test_ds = FacialKeypointsDataset(test_df.reset_index(drop=True))
train_dl = DataLoader(train_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)

def get_model():
  model = models.vgg16(pretrained=True)
  for params in model.parameters():
    params.requires_grad = False
  model.avgpool = nn.Sequential(nn.Conv2d(512, 512, 3), nn.MaxPool2d(2), nn.Flatten())
  model.classifier = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 136), nn.Sigmoid())
  criterion = nn.L1Loss()
  optimizer = Adam(model.parameters(), lr=1e-4)
  return model, criterion, optimizer

def train_batch(img, kps, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    loss.backward()
    optimizer.step()
    return loss

def validate_batch(img, kps, model, criterion):
    model.eval()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    return _kps, loss

model, criterion, optimizer = get_model()

train_loss, test_loss = [], []
n_epochs = 50
for epoch in range(n_epochs):
  print(f" epoch {epoch+ 1} : 50")
  epoch_train_loss, epoch_test_loss = 0, 0
  for ix, (img,kps) in enumerate(train_dl):
    loss = train_batch(img, kps, model, optimizer, criterion)
    epoch_train_loss += loss.item()
  epoch_train_loss /= (ix+1)
  for ix,(img,kps) in enumerate(test_dl):
    ps,  loss = validate_batch(img, kps, model, criterion)
    epoch_test_loss += loss.item()
  epoch_test_loss /= (ix+1)
  train_loss.append(epoch_train_loss)
  test_loss.append(epoch_test_loss)

# Commented out IPython magic to ensure Python compatibility.
epochs = np.arange(50)+1
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# %matplotlib inline
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, test_loss, 'r', label='Test loss')
plt.title('Training and Test loss over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.show()