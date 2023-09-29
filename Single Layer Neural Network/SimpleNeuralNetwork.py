# XOR Problem -> Classification Problem

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# print(x.shape)
# print(y.shape)



model = Sequential()

model.add(Dense(4, input_dim=2, activation='sigmoid')) #hidden layer
model.add(Dense(1, input_dim=4, activation='sigmoid')) #output layer

print(model.weights)

model.compile(loss="mean_squared_error", optimizer="adam", metrics=['binary_accuracy'])
model.fit(x, y, epochs=10000, verbose=2)

print("Prediction After Training")
print(model.predict(x))











