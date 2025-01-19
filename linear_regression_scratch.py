import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LinearRegression():
    def fit(self, feature, label, learning_rate, epochs):
        self.m = len(feature)
        self.weight = np.zeros((feature.shape[1], 1))
        self.base = 0
        for i in range(epochs):
            if i % 1000 == 0:
                print(f'epoch {i}, loss: {self.compute_loss(feature, label)}')
            self.update_weights(feature, label, learning_rate, i)
        return self

    def update_weights(self, feature, label, learning_rate, i):
        y_pred_array = self.predict(feature)
        diff = label - y_pred_array
        loss_derivative_weight = - (2 * np.dot(feature.T, diff)) / self.m
        loss_derivative_base = - (2 * np.sum(diff)) / self.m
        self.weight = self.weight - learning_rate * loss_derivative_weight
        self.base = self.base - learning_rate * loss_derivative_base
        return self
        
    def predict(self, feature):
        return np.dot(feature, self.weight) + self.base
    
    def compute_loss(self, feature, label):
        y_pred = self.predict(feature)
        return np.mean((label - y_pred) ** 2)

def plot_prediction(X_test, Y_test, Y_pred):
    plt.scatter(X_test, Y_test, color='blue')
    plt.plot(X_test, Y_pred, color='red')
    plt.title('Taxi miles vs fares')
    plt.xlabel('Miles')
    plt.ylabel('Fares')
    plt.show()
  
def build_model():
    data_url = 'https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv'
    taxi_rate_data = pd.read_csv(data_url)
    feature = taxi_rate_data[['TRIP_MILES']].values
    label = taxi_rate_data[['FARE']].values
    X_train, X_test, Y_train, Y_test = train_test_split(feature, label, test_size=0.2, random_state=0) 
    learning_rate = 0.01
    epochs = 10000
    model = LinearRegression()
    model.fit(X_train, Y_train, learning_rate, epochs)
    Y_pred = model.predict(X_test)
    plot_prediction(X_test, Y_test, Y_pred)

build_model()

    



