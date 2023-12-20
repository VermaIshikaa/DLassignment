from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups

# Load the 20 newsgroups dataset for training and testing
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# Create a text classification pipeline using CountVectorizer, TfidfTransformer, and GradientBoostingClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', GradientBoostingClassifier(n_estimators=50, verbose=2)),
                     ])

# Fit the pipeline on the training data
text_clf.fit(X_train, y_train)

# Make predictions on the test data
predicted = text_clf.predict(X_test)

# Print the classification report, which includes precision, recall, and F1-score
print(metrics.classification_report(y_test, predicted))
