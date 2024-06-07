import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from model_training_script import ModelTraining

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        # Setup for the tests
        self.data_path = './mle-intv-main/data/train.csv'
        self.model_path = './mle-intv-main/test_model.pkl'
        self.trainer = ModelTraining(data_path=self.data_path, model_path=self.model_path)
        self.trainer.load_data()

    def test_load_data(self):
        self.assertIsNotNone(self.trainer.df, "Dataframe should not be None after loading data")
        self.assertGreater(len(self.trainer.df), 0, "Dataframe should not be empty after loading data")

    def test_preprocess_data(self):
        df_X, df_label = self.trainer.preprocess_data()
        self.assertIsNotNone(df_X, "Feature dataframe should not be None after preprocessing data")
        self.assertIsNotNone(df_label, "Label series should not be None after preprocessing data")

    def test_train_model(self):
        df_X, df_label = self.trainer.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_label, random_state=1337)
        self.trainer.train_model(X_train, y_train)
        self.assertIsNotNone(self.trainer.model, "Model should not be None after training")

    def test_save_model(self):
        df_X, df_label = self.trainer.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_label, random_state=1337)
        self.trainer.train_model(X_train, y_train)
        self.trainer.save_model()
        import os
        self.assertTrue(os.path.exists(self.model_path), "Model file should exist after saving")

if __name__ == '__main__':
    unittest.main()
