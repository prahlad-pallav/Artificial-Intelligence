from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

# print(digits)

# print(digits.data.shape)

x_digits = digits.data
y_digits = digits.target

# print(y_digits.shape)

# number of features after the PCA -> n_components
estimator = PCA(n_components=2)

x_pca = estimator.fit_transform(x_digits)

# print(x_pca)

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

for i in range(len(colors)):
    px = x_pca[:, 0][y_digits == i]
    py = x_pca[:, 1][y_digits == i]
    plt.scatter(px, py, c=colors[i])
    plt.legend(digits.target_names)

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# explained variance shows how much information can be attributed to the principal components

print("Explianed variance: %s" %estimator.explained_variance_ratio_)


plt.show()












