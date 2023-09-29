import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import math

credit_data = pd.read_csv("credit_data.csv")

# print(credit_data.head())

# print(credit_data.describe())

# print(credit_data.corr())

features = credit_data[["income", "age", "loan"]]
target = credit_data.default

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = LogisticRegression()
model.fit = model.fit(feature_train, target_train)

print(model.fit.coef_)

predictions = model.fit.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))