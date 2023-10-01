import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# XOR -> it is a non-linearly separable problem -> require hidden layer
# for linearly separable problem -> no hidden layer required
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")

target_data = np.array([[0], [1], [1], [0]], "float32")

model = Sequential()


# first layer has 2 neurons -> 2 input features

model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# output layer has 1 neuron -> 1 output data

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

# epochs -> no of iteration over the entire dataset
# verbose -> 0 is silent, 1 and 2 are showing results

model.fit(training_data, target_data, epochs=500, verbose=2)

print(model.predict(training_data).round())