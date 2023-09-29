import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
import math

credit_data = pd.read_csv("credit_data.csv")

features = credit_data[["income", "age", "loan"]]
target = credit_data.default

x = np.array(features).reshape(-1, 3)
y = np.array(target)

model = LogisticRegression()
predicted = cross_validate(model, x, y, cv=5)

print(predicted['test_score'])
print(np.mean(predicted['test_score']))