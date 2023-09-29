from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=0)

# print(training_data)
# print(training_data.target)

# print("\n".join(training_data.data[0].split("\n")[:30]))
# print("Target is: ", training_data.target_names[training_data.target[1]])
#
# print(training_data.data)

count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(training_data.data)
# print(count_vector.vocabulary_)


tfid_transformer = TfidfTransformer()
x_train_tfidf = tfid_transformer.fit_transform(x_train_counts)
# print(x_train_tfidf)


model = MultinomialNB().fit(x_train_tfidf, training_data.target)

new = ['My favorite topic is something to do with Religion', 'This has nothing to do with church or religion', 'Software Engineering is getting hotter and hotter nowadays']

x_new_counts = count_vector.transform(new)

x_new_tfidf = tfid_transformer.transform(x_new_counts)

predicted = model.predict(x_new_tfidf)

print(predicted)

for doc, category in zip(new, predicted):
    print('%r ====> %s' % (doc, training_data.target_names[category]))