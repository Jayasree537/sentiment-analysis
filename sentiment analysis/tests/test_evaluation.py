import unittest
from sklearn.metrics import accuracy_score
from src import data_preprocessing, model_training, sentiment_analysis, evaluation

class TestEvaluation(unittest.TestCase):
    
    def test_evaluate_model(self):
        data, labels = data_preprocessing.load_data()
        X_train, X_test, y_train, y_test = data_preprocessing.split_data(data, labels)
        vectorizer, clf = model_training.train_model(X_train, y_train)
        y_pred = sentiment_analysis.predict_sentiment(vectorizer, clf, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

if __name__ == '__main__':
    unittest.main()
