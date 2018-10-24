

# -*- coding: utf-8 -*-
import numpy as np
from common import logger
from general import *
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn import manifold, datasets

import show_tsne
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin


# def create_augmented_train(X, y, model, test, features, target, sample_rate):
#     """
#     Create and return the augmented_train set that consists
#     of pseudo-labeled and labeled data.
#     """
#     num_of_samples = int(len(test) * sample_rate)

#     # Train the model and creat the pseudo-labeles
#     model.fit(X, y)
#     pseudo_labeles = model.predict(test[features])

#     # Add the pseudo-labeles to the test set
#     augmented_test = test.copy(deep=True)
#     augmented_test[target] = pseudo_labeles

#     # Take a subset of the test set with pseudo-labeles and append in onto
#     # the training set
#     sampled_test = augmented_test.sample(n=num_of_samples)
#     temp_train = pd.concat([X, y], axis=1)
#     augemented_train = pd.concat([sampled_test, temp_train])

#     # Shuffle the augmented dataset and return it
#     return shuffle(augemented_train)


class PseudoLabeler(BaseEstimator, RegressorMixin):
    def __init__(self, model, test, features, target, sample_rate=0.2, seed=42):
        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed

        self.test = test  # unlabeled data
        self.features = features
        self.target = target

    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "test": self.test,
            "features": self.features,
            "target": self.target,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        if self.sample_rate > 0.0:
            augemented_train = self.__create_augmented_train(X, y)
            self.model.fit(
                augemented_train[self.features], augemented_train[self.target]
            )
        else:
            self.model.fit(X, y)

        return self

    def __create_augmented_train(self, X, y):
        num_of_samples = int(len(self.test) * self.sample_rate)

        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.test[self.features])

        # Add the pseudo-labels to the test set
        augmented_test = self.test.copy(deep=True)
        augmented_test[self.target] = pseudo_labels

        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        sampled_test = augmented_test.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        augemented_train = pd.concat([sampled_test, temp_train])

        return shuffle(augemented_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_model_name(self):
        return self.model.__class__.__name__


if __name__ == "__main__":
    """
    ######################  psuedo label #################################################
    """
    # Preprocess the data
    train = pd.read_csv(
        # "/home/tq/zgq/zdata/crop-class/zgq/reduced/mississipi-allhard-reduced.csv"
        "/home/tq/zgq/zdata/crop-class/zgq/mississipi-allhard-full.csv"
    )
    test = pd.read_csv(
        # "/home/tq/zgq/zdata/crop-class/zgq/reduced/mississipi-soft-reduced.csv"
        "/home/tq/zgq/zdata/crop-class/zgq/mississipi-soft-full.csv"
    )
    vali = pd.read_csv(
        # "/home/tq/zgq/zdata/crop-class/zgq/reduced/mississipi-soft-label-reduced.csv"
        "/home/tq/zgq/zdata/crop-class/zgq/mississipi-soft-label-full.csv"
    )
    print(train.shape, test.shape, vali.shape)
    print(train.head())

    features = train.columns[2:]
    print("tests:")
    print(test.head())
    target = "y"

    X_train, X_test = train[features], test[features]
    y_train = train[target]
    y_vali = vali[target]
    y_vali = np.array(y_vali.tolist())

    model = PseudoLabeler(XGBRegressor(), test, features, target)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_r = y_pred
    for i in range(y_pred.shape[0]):
        if y_pred[i] > 3 and y_pred[i] < 4.5:
            y_pred_r[i] = 3
        elif y_pred[i] > 4.5 and y_pred[i] < 6:
            y_pred_r[i] = 6
        elif y_pred[i] > 6:
            y_pred_r[i] = 6
        elif y_pred[i] < 0:
            y_pred_r[i] = 0
        else:
            y_pred_r[i] = round(y_pred[i])

    cm = confusion_matrix(y_vali, y_pred_r)
    cm_plot(y_vali, y_pred_r).show()

    print(
        classification_report(
            y_vali,
            y_pred_r,
            labels=None,
            target_names=None,
            sample_weight=None,
            digits=2,
        )
    )
