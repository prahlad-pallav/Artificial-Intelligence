import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam


iris_data = load_iris()

features = iris_data.data

# print(features)

targets = iris_data.target

labels = targets.reshape(-1, 1)

# print(labels)

# print(targets)

# we have 3 classes so the labels will have 3 classes
# first class : (1, 0, 0) , second class : (0, 1, 0), third class : (0, 0, 1)
# neural networks like the value range between o and 1

encoder = OneHotEncoder()
targets = encoder.fit_transform(labels).toarray()

# print(targets)

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

model = Sequential()

# hidden layer
model.add(Dense(10, input_dim=4, activation='sigmoid'))
#output layer
model.add(Dense(3, activation='softmax'))

# we can define the loss function MSE or negative log likelihood
# optimizer will find the right adjustments for the weights, SGD, Adagrad, ADAM.





