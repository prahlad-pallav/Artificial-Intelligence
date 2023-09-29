import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import preprocessing

data = pd.read_csv("wine.csv", sep=";")

def is_tasty(quality):
    if quality >= 7:
        return 1
    else:
        return 0

# print(data)

features = data[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH"
                 , "sulphates", "alcohol"]]

# print(features)

data['tasty'] = data['quality'].apply(is_tasty)

target = data['tasty']

# print(target)
#
# print(data)

x = np.array(features).reshape(-1, 11)
y = np.array(target)

# print(x)
# print(y)


x = preprocessing.MinMaxScaler().fit_transform(x)
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)


# print(x)


param_dist = {
    'n-estimators': [10, 50, 200],
    'learning_rate': [0.01, 0.05, 0.3, 1]
}

estimator = AdaBoostClassifier()

grid_search = GridSearchCV(estimator=estimator, param_grid=param_dist, cv=10)
grid_search.fit(feature_train, target_train)

grid_predictions = grid_search.predict(feature_test)

print(confusion_matrix(target_test, grid_predictions))
print(accuracy_score(target_test, grid_predictions))