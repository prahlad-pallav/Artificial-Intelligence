from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()


tfidf = vec.fit_transform(['In like Machine Learning and Clustering algorithms',
                           'Apples, oranges and any kind of fruits are healthy',
                           'Is it feasible with machine learning algorithms?',
                           'My family is happy because of the healthy fruits'])

# document term matrix
# print(tfidf)
# print(tfidf.A)
# print(vec.get_feature_names_out())
# similarity matrix
print((tfidf*tfidf.T).A)
