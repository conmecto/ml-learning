import torch.nn as nn

# Applied as end layers might loose original input features and in backpropagation init layer might have vanishing grad
class ResLayer(nn.Module):
  def __init__(self, ni, no, kernel_size, stride=1):
    super(ResLayer, self).__init__()
    padding = kernel_size - 2
    self.conv = nn.Sequential(
        nn.Conv2d(ni, no, kernel_size, stride, padding=padding),
        nn.ReLU(),
    )

  def forward(self, x):
    y = self.conf(x) + x
    return y