import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

#  logistic regression -> 93%
#  KNN -> 97.5% (with Data Normalisation) and 84% (without Data Normalisation)
#  Random Forest -> 99%
#  Deep Neural Networks -> More than 99%

credit_data = pd.read_csv("credit_data.csv")


features = credit_data[["income", "age", "loan"]]
targets = credit_data.default

x = np.array(features).reshape(-1, 3)
y = np.array(targets)

# feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.3)

model = RandomForestClassifier()

predicted = cross_validate(model, x, y, cv=10)
print(np.mean(predicted['test_score']))




