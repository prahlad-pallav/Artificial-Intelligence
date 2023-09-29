import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import datasets

x, y = datasets.make_moons(n_samples=1500, noise=0.05)

# print(x)
# print(y)

x1 = x[:, 0]
x2 = x[:, 1]

plt.scatter(x1, x2, s=5)

# plt.show()

dbscan = DBSCAN(eps=0.3)
dbscan.fit(x)

y_pred = dbscan.labels_.astype(np.int_)

# print(y_pred)

colors = np.array(['#ff0000', '#00ff00'])

plt.scatter(x1, x2, s=5, color=colors[y_pred])

# plt.show()

kmeans = KMeans(n_clusters=2)

kmeans.fit(x)

y_pred = kmeans.labels_.astype(np.int_)

plt.scatter(x1, x2, s=5, color=colors[y_pred])

plt.show()