import torch 
import torch.nn as nn
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt


# t_data = torch.ones(5)
# n_data = t_data.numpy()

# print(id(id(t_data)))
# print(id(id(n_data)))
# t_data.add_(1)
# print(t_data)
# print(n_data)

# np_data = np.ones(5)
# torch_data = torch.from_numpy(np_data)

# np.add(np_data, 1, out=np_data)
# print(np_data)
# print(torch_data)

# n = [[1, 2], [3, 4]]
# z = torch.tensor(n)
# shape = (2, 3)
# y = torch.rand(shape)
# print(y)
# x = torch.rand_like(z, dtype=torch.float)
# print(x)

# print(x.shape)
# print(y.dtype)
# print(y.device)

# print(torch.cat([x, y], dim=1))

# print(x.matmul(torch.ones((2, 2))))


# x = torch.randint(0, 10, (2, 2, 2))
# print(x)
# print(x.max(dim=1))

# x = torch.randn((2, 3), requires_grad=True)

# print(x)

# y = x.pow(2)
# z = y.sum()

# print(z)

# print(z.backward())
# print(x.grad)

device = 'cpu'

X = [[1, 2], [3, 4], [4, 5], [6, 7]]
Y = [[3], [7], [9], [13]]

X = torch.tensor(X).float()
Y = torch.tensor(Y).float()

class MyNeuralNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(2, 8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8, 1)

    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x
    
myNet = MyNeuralNet().to(device)

# for par in myNet.parameters():
#     print(par)

loss_func = nn.MSELoss()

# Y_pred = myNet.forward(X)
# print(Y_pred)
# loss_val = loss_func(Y_pred, Y)
# print(loss_val)

opt = SGD(myNet.parameters(), lr=0.001)

loss_history = []

for _ in range(50):
    opt.zero_grad()
    Y_pred = myNet.forward(X)
    print('Y_pred', Y_pred)
    loss_val = loss_func(Y_pred, Y)
    loss_val.backward()
    opt.step()
    loss_history.append(loss_val.item())

plt.plot(loss_history)
plt.title('Loss variation over increasing epochs')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.show()


