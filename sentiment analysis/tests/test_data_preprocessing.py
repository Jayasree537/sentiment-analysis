import unittest
from src import data_preprocessing

class TestDataPreprocessing(unittest.TestCase):
    
    def test_load_data(self):
        data, labels = data_preprocessing.load_data()
        self.assertEqual(len(data), len(labels))
        self.assertGreater(len(data), 0)
        self.assertGreater(len(labels), 0)
        self.assertIn('happy', labels)
        self.assertIn('sad', labels)
        self.assertIn('neutral', labels)

    def test_split_data(self):
        data, labels = data_preprocessing.load_data()
        X_train, X_test, y_train, y_test = data_preprocessing.split_data(data, labels)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)

if __name__ == '__main__':
    unittest.main()
