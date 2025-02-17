import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

x = [[1,2],[3,4],[5,6],[7,8]]
y = [[3],[7],[11],[15]]

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
# class SimpleNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.input_to_hidden_layer = nn.Linear(2, 8)
#         self.hidden_layer_activation = nn.ReLU()
#         self.hidden_to_output_layer = nn.Linear(8, 1)

#     def forward(self, x):
#         hidden1 = self.input_to_hidden_layer(x)
#         hidden2 = self.hidden_layer_activation(hidden1)
#         output = self.hidden_to_output_layer(hidden2)
#         return output

ds = MyDataset(x, y)
dl = DataLoader(ds, batch_size=2, shuffle=True)

# myNet = SimpleNN()

model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))

loss_func = nn.MSELoss()

opt = SGD(model.parameters(), lr=0.001)

loss_history = []
for _ in range(1000):
    for data in dl:
        current_x, current_y = data
        opt.zero_grad()
        loss_value = loss_func(model(current_x), current_y)
        loss_value.backward() 
        opt.step()
        loss_history.append(loss_value.item())


# saving the model
save_path = 'mymodel.pth'
print('model.state_dict()', model.state_dict())
torch.save(model.state_dict(), save_path)

# plt.plot(loss_history)
# plt.title('Loss variation over increasing epochs')
# plt.xlabel('epochs')
# plt.ylabel('loss value')
# plt.show()

# loading the state of the saved model 
load_path = 'mymodel.pth'
new_model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
new_model.load_state_dict(torch.load(load_path))

test_x = [[10, 11], [110, 11]]
test_x = torch.tensor(test_x).float()
print('test_x', test_x)
print(new_model(test_x))