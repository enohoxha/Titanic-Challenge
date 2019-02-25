import pandas as pd
import input.feature_engineering as fe
import matplotlib.pyplot as plt
import re
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

pd.set_option('display.width', 1000)


class DataSet:

    def __init__(self, train_path, test_path):
        self.i = 0

        # Get train data
        train_data = pd.read_csv(train_path)
        self.train_df = fe.feature_extraction(train_data, True)

        # Get test data
        test_data = pd.read_csv(test_path)
        self.test_df = fe.feature_extraction(test_data, False)

        # Declare variables

        self.train = None
        self.train_labels = None

        self.test = None
        self.test_labels = None

        self.production = None

    def load_data(self):
        train_set, test_set = self.train_test_split(self.train_df)

        self.train_labels = self.one_hot_encode(train_set.Survived.values)
        self.test_labels = self.one_hot_encode(test_set.Survived.values)

        train_set = train_set.drop("Survived", axis=1)
        test_set = test_set.drop("Survived", axis=1)
        prod_set = self.test_df[['Pclass', 'Sex',  'Embarked', 'Age_Range', 'Family_Count']]

        scale = MinMaxScaler()
        self.train = scale.fit_transform(train_set)
        self.test = scale.transform(test_set)
        self.production = scale.transform(prod_set)

    @staticmethod
    def train_test_split(dataset, split_size=150):
        return dataset.head(dataset.shape[0] - split_size), dataset.tail(split_size)

    @staticmethod
    def one_hot_encode(labels):
        b = np.zeros((len(labels), 2))
        b[np.arange(len(labels)), labels] = 1
        return b

    @staticmethod
    def rev_one_hot_encode(labels):
        arr = [np.argmax(y, axis=None, out=None) for y in labels]
        return arr

    def next_batch(self, batch_size):
        train_data = self.train[self.i: self.i + batch_size]
        train_labels = self.train_labels[self.i: self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.train)
        return train_data, train_labels
