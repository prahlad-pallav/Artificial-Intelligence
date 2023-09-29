from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

mnist_data = fetch_openml('mnist_784')

features = mnist_data.data
targets = mnist_data.target

print(features.shape)
print(targets.shape)

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.15)

scalar = StandardScaler()

scalar.fit(feature_train)

feature_train = scalar.transform(feature_train)
feature_test = scalar.transform(feature_test)

# we keep 95% variance -> so 95% of the original information
pca = PCA(.95)

pca.fit(feature_train)

feature_train = pca.transform(feature_train)
feature_test = pca.transform(feature_test)

print(feature_train.shape)



