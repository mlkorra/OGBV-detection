from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
 
models = {
    "lr": LogisticRegression(),
    "nb": naive_bayes.MultinomialNB()
}

vecs = {
    "cv" : CountVectorizer(),
    "tfidf" : TfidfVectorizer()
}

