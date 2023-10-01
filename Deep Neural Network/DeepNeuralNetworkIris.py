import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

dataset = load_iris()

features = dataset.data
# print(features)

targets = dataset.target

y = targets.reshape(-1, 1)
# print(y)

encoder = OneHotEncoder(sparse=False)
targets = encoder.fit_transform(y)

print(targets)

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

model = Sequential()

# first layer has 4 neurons -> 4 input features
# relu -> vanicing gradient problem
# softmax -> classification problem for many
# sigmoid -> classification problem for 2

model.add(Dense(15, input_dim=4, activation='relu'))
model.add(Dense(15, input_dim=15, activation='relu'))
model.add(Dense(15, input_dim=15, activation='relu'))
model.add(Dense(3, activation='softmax'))

optimizer = Adam(learning_rate=0.015)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# cross entropy has something to do with information theory

model.fit(feature_train, target_train, epochs=1000, batch_size=20, verbose=2)

results = model.evaluate(feature_test, target_test)

print("Accuracy on the test dataset: %.2f" % results[1])



















