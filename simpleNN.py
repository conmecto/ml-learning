import numpy as np

class MySimpleNN:
    def __init__(self, weights):
        self.weights = weights

    def mse(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        self.h = np.dot(x, self.weights[0]) + self.weights[1]
        self.a = self.sigmoid(self.h)
        self.z = np.dot(self.a, self.weights[2]) + self.weights[3]
        self.y_pred = self.sigmoid(self.z)
        return self.y_pred

    def backward(self, x, y, learning_rate):
        d_loss_y_pred = self.y_pred - y
        d_y_pred_z = self.sigmoid_derivative(self.y_pred)
        d_loss_z = d_loss_y_pred * d_y_pred_z

        #Gradients for output layer's weights and bias
        d_loss_w2 = np.dot(self.a.T, d_loss_z) 
        d_loss_w3 = np.sum(d_loss_z, axis=0, keepdims=True)

        #Backpropogate to hidden layer 
        d_z_a = self.weights[2]
        d_loss_a = np.dot(d_loss_z, d_z_a.T)
        d_a_h = self.sigmoid_derivative(self.a)
        d_loss_h = d_loss_a * d_a_h

        #Gradients for hidden layer's weights and bias
        d_loss_w0 = np.dot(x.T, d_loss_h)
        d_loss_w1 = np.sum(d_loss_h, axis=0, keepdims=True)

        # Update weights and biases
        self.weights[3] -= learning_rate * d_loss_w3
        self.weights[2] -= learning_rate * d_loss_w2
        self.weights[1] -= learning_rate * d_loss_w1
        self.weights[0] -= learning_rate * d_loss_w0

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.mse(y_pred, y)
            self.backward(X, y, learning_rate)
            print(f'Epoch {epoch}, Loss: {loss}')


# XOR 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
y = np.array([[0], [1], [1], [0]])  
weights = []
weights.append(np.random.randn(2, 3))
weights.append(np.zeros((1, 3)))
weights.append(np.random.randn(3, 1))
weights.append(np.zeros((1, 1)))
nn = MySimpleNN(weights)
nn.train(X, y, epochs=1000, learning_rate=0.01)