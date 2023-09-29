import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  cross_validate
from sklearn.metrics import confusion_matrix
from sklearn import datasets


iris_data = datasets.load_iris()

features = iris_data.data
targets = iris_data.target

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

# model = DecisionTreeClassifier(criterion='entropy')
model = DecisionTreeClassifier(criterion='gini')

predicted = cross_validate(model, features, targets, cv=10)

print(np.mean(predicted['test_score']))

