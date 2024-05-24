import unittest
from src import data_preprocessing, model_training, sentiment_analysis

class TestSentimentAnalysis(unittest.TestCase):
    
    def test_predict_sentiment(self):
        data, labels = data_preprocessing.load_data()
        X_train, X_test, y_train, y_test = data_preprocessing.split_data(data, labels)
        vectorizer, clf = model_training.train_model(X_train, y_train)
        y_pred = sentiment_analysis.predict_sentiment(vectorizer, clf, X_test)
        self.assertEqual(len(y_pred), len(X_test))

if __name__ == '__main__':
    unittest.main()
