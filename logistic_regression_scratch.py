import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(X, weight, base):
    z = np.dot(X, weight) + base
    return 1 / (1 + np.exp(-z))

def loss(h, y, m):
    l1 = - np.dot(y.T, np.log(h))
    l2 = - np.dot((1 - y).T, np.log(1 - h))
    return (l1 + l2) / m

def gradient_descent(X, h, y, m):
    diff = h - y
    dot_product = np.dot(X.T, diff)
    dW = dot_product / m
    dB = np.sum(diff) / m
    return dW, dB

data = pd.read_csv("../input/telecom-dataset/telco.csv")
#.iloc[:100]
print("Dataset size")
print("Rows {} Columns {}".format(data.shape[0], data.shape[1]))


telco_data = data[['tenure', 'MonthlyCharges', 'Churn']]
learning_rate = 0.001
epochs = 100000
X_values = telco_data.iloc[:,:-1].values 
Y_values_str = telco_data.iloc[:,-1].values.reshape(-1, 1)
Y_values = np.where(Y_values_str == 'Yes', 1, 0)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_values, Y_values, test_size=0.2, random_state=0) 

m = X_train.shape[0]
n = X_train.shape[1]

weight = np.zeros((n, 1))
base = 0

for i in range(epochs):
    h = sigmoid(X_train, weight, base)
    dW, dB = gradient_descent(X_train, h, Y_train, m)
    weight = weight - learning_rate * dW
    base = base - learning_rate * dB
    
    if i % 5000 == 0:
        print(f'epoch: {i}')
        print(f'loss: {loss(h, Y_train, m)}')

result = sigmoid(X_test, weight, base)

# print(Y_test)
# print(result)
x_monthly_charges = X_test[:,0].reshape(-1, 1)
plt.scatter(x_monthly_charges[Y_test == 0], Y_test[Y_test == 0], color='blue')
plt.scatter(x_monthly_charges[Y_test == 1], Y_test[Y_test == 1], color='red')
plt.plot(x_monthly_charges, result, color='green')
plt.xlabel('MonthlyCharges')
plt.ylabel('Probability')
plt.title('1D Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()            