import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

house_data = pd.read_csv("house_prices.csv")
# print(house_data)

size = house_data['sqft_living']
# print(size)

price = house_data['price']
# print(price)

# The final outcome of the conversion is that the number of elements in the final array
# is same as that of the initial array or data frame.

# -1 corresponds to the unknown count of the row or column.

x = np.array(size).reshape(-1, 1)
# print(x)

y = np.array(price).reshape(-1, 1)
# print(y)

model = LinearRegression()
model.fit(x, y)

regression_model_mse = mean_squared_error(x, y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R Squared Value: ", model.score(x, y))
print("Model Coefficient: ", model.coef_[0])
print("Model Intercept: ", model.intercept_[0])
print("Prediction by the model", model.predict([[1000]]))

plt.scatter(x, y, color="green")
plt.plot(x, model.predict(x), color="black")
plt.title("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()




