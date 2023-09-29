import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import cross_validate


cancer_data = datasets.load_breast_cancer()

# print(cancer_data.data)
# print(cancer_data.target)


features = cancer_data.data
labels = cancer_data.target

# print(features.shape)

feature_train, feature_test, target_train, target_test = train_test_split(features, labels, test_size=0.3)

# model = DecisionTreeClassifier(max_depth=3)
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)

predicted = cross_validate(model, features, labels, cv=10)

print(np.mean(predicted['test_score']))