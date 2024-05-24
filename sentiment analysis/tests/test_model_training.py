import unittest
from src import data_preprocessing, model_training

class TestModelTraining(unittest.TestCase):
    
    def test_train_model(self):
        data, labels = data_preprocessing.load_data()
        X_train, X_test, y_train, y_test = data_preprocessing.split_data(data, labels)
        vectorizer, clf = model_training.train_model(X_train, y_train)
        self.assertIsNotNone(vectorizer)
        self.assertIsNotNone(clf)

if __name__ == '__main__':
    unittest.main()
