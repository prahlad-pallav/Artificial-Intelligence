import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

iris_data = datasets.load_iris()

features = iris_data.data
targets = iris_data.target

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.3)

model = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=123)

fitted_model = model.fit(feature_train, target_train)

predictions = fitted_model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))

