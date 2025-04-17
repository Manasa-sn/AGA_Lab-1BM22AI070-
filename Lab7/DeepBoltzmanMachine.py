import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

data = fetch_california_housing()
X = data.data
y = data.target


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

class DBM:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()

        # Add the first hidden layer
        model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))
        model.add(layers.Dense(self.layer_sizes[0], activation='relu'))

        for size in self.layer_sizes[1:]:
            model.add(layers.Dense(size, activation='relu'))

        model.add(layers.Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def evaluate(self, X_test, y_test):
        loss = self.model.evaluate(X_test, y_test)
        return loss

    def predict(self, X):
        return self.model.predict(X)

dbm = DBM(layer_sizes=[256, 128, 64])

dbm.train(X_train, y_train, epochs=50, batch_size=32)

test_loss = dbm.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss}")

y_pred = dbm.predict(X_test)

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Housing Prices')
plt.show()
