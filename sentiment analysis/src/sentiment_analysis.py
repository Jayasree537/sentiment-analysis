def predict_sentiment(vectorizer, clf, X_test):
    X_test_counts = vectorizer.transform(X_test)
    return clf.predict(X_test_counts)

