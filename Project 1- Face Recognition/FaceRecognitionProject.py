from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics


olivetti_data = fetch_olivetti_faces()

# there are 400 images - 10x40 (40 people - 1 person has 10 images) - 1 image = 64x64 pixels
features = olivetti_data.data
# we represent target variables (people) with integers (face ids)
targets = olivetti_data.target


# pixel intensities are normalised -> no need for min-max normalisation

# print(features)
# print(targets)

# fig, sub_plots = plt.subplots(nrows=5, ncols=8, figsize=(14, 8))
#
# sub_plots = sub_plots.flatten()
# # print(sub_plots)
#
# for unique_user_id in np.unique(targets):
#     image_index = unique_user_id * 10
#     sub_plots[unique_user_id].imshow(features[image_index].reshape(64, 64), cmap='gray')
#     sub_plots[unique_user_id].set_xticks([])
#     sub_plots[unique_user_id].set_yticks([])
#     sub_plots[unique_user_id].set_title("Face Value: %s" % unique_user_id)
#
# plt.suptitle("The data of 40 different people")
# plt.show()


# 10 different images of same person

# fig, sub_plots = plt.subplots(nrows=1, ncols=10, figsize=(18, 9))
#
# for j in range(10):
#     sub_plots[j].imshow(features[j].reshape(64, 64), cmap='gray')
#     sub_plots[j].set_xticks([])
#     sub_plots[j].set_yticks([])
#     sub_plots[j].set_title("Face id=0")

# plt.show()


# split the original data-set (training and test set)
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, stratify=targets, random_state=0)

# From 4096 to 100 -> We want to maximize the explained variance

# pca = PCA()
# pca.fit(features)
#
# plt.figure(1, figsize=(12, 8))
# plt.plot(pca.explained_variance_, linewidth=2)
# plt.xlabel("Features")
# plt.ylabel("Explained Variance")
#
# plt.show()


# let's try to find the optimal number of eigenvectors (principle components)
pca = PCA(n_components=100, whiten=True)
pca.fit(X_train)
X_pca = pca.fit_transform(features)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# print(features.shape)
# print(X_train_pca.shape)


# after we find the optimal 100 PCA numbers we can check the "eigenvalues"
# 1 principal component or eigenvector has 4096 features
number_of_eigenfaces = len(pca.components_)

eigen_faces = pca.components_.reshape((number_of_eigenfaces, 64, 64))

fig, subplots = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
# print(subplots)
subplots = subplots.flatten()
# print(subplots)


for i in range(number_of_eigenfaces):
    subplots[i].imshow(eigen_faces[i], cmap="gray")
    subplots[i].set_xticks([])
    subplots[i].set_yticks([])

plt.suptitle("EigenFaces")
# plt.show()



# eigenfaces = eigenvector (when face recognition is applicable)
# print(number_of_eigenfaces)



# let's use the machine learning models

models = [("Logistic Regression", LogisticRegression()), ("Support Vector Machine", SVC()), ("Naive Bayes Classifier", GaussianNB())]



for name, model in models:

    classifier_model = model
    classifier_model.fit(X_train_pca, y_train)

    y_predicted = classifier_model.predict(X_test_pca)
    print("Results with %s" % name)
    print("Accuracy score: %s" % (metrics.accuracy_score(y_test, y_predicted)))

# for name, model in models:
#
#     kfold = KFold(n_splits=5, shuffle=True, random_state=0)
#     cv_scores = cross_val_score(model, X_pca, targets, cv=kfold)
#     print("Mean of the cross-validation scores: %s" % cv_scores.mean())
