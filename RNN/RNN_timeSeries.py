import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout

NUM_OF_PREV_ITEMS = 5

def reconstruct_data(data_set, n=1):
    x, y = [], []

    for i in range(len(data_set) - n -1):
        a = data_set[i :(i + n), 0]
        x.append(a)
        y.append(data_set[i + n, 0])

    return numpy.array(x), numpy.array(y)

# we need to make sure the results will be the same every time, we execute the algorithm

numpy.random.seed(1)

data_frame = read_csv('daily_min_temperatures.csv', usecols=[1])

# print(data_frame)
# print(data_frame.values)
#
# plt.plot(data_frame)
# plt.show()


data = data_frame.values

data = data.astype('float32')

# min-max normalisation -> transfer data in the range [0,1]
scalar = MinMaxScaler(feature_range=(0, 1))

data = scalar.fit_transform(data)
# print(data)


# split the datasets into train and test sets -> 70% and 30%

train, test = data[0: int(len(data) * 0.7), :], data[int(len(data) * 0.7): len(data), :]

train_x, train_y = reconstruct_data(train, NUM_OF_PREV_ITEMS)
test_x, test_y = reconstruct_data(test, NUM_OF_PREV_ITEMS)
# print(train_x)
# print(train_y)

# we need to reshape input to be [numOfSamples, timeSteps, numOfFeatures]
# here time steps is 1 because we want to predict the next value (t+1)

# print((train_x.shape[0], 1, train_x.shape[1]))

train_x = numpy.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = numpy.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

# print(train_x)

# create the LSTM model

model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape =(1, NUM_OF_PREV_ITEMS)))
model.add(Dropout(0.5))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=50))
model.add(Dropout(0.3))
model.add(Dense(units=1))

# optimize the model with ADAM optimizer -> here we're dealing with the regression problem

model.compile(loss='mean_squared_error', optimizer='Adam')
model.fit(train_x, train_y, epochs=10, batch_size=16, verbose=2)

# make predictions and min-max normalisation

test_predict = model.predict(test_x)
test_predict = scalar.inverse_transform(test_predict)
test_labels = scalar.inverse_transform([test_y])

# print(test_labels)

test_score = mean_squared_error(test_labels[0], test_predict[:, 0])
print('Score on test set: %.2f MSE' % test_score)

# plot the results (original data + predictions)

test_predict_plot = numpy.empty_like(data)
test_predict_plot[:, :] = numpy.nan
test_predict_plot[len(train_x) + 2*NUM_OF_PREV_ITEMS+1 : len(data)-1, :] = test_predict

print(test_predict_plot)
plt.plot(scalar.inverse_transform(data))
plt.plot(test_predict_plot, color="green")
plt.show()





