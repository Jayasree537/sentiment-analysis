import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    clf = MultinomialNB()
    clf.fit(X_train_counts, y_train)
    
    # Save the vectorizer and classifier
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    joblib.dump(clf, 'models/classifier.pkl')
    
    return vectorizer, clf

