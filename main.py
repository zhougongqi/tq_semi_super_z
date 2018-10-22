# -*- coding: utf-8 -*-
import numpy as np
from common import logger
from general import *
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn import manifold, datasets

import show_tsne
from PseudoLabeler import PseudoLabeler
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.decomposition import PCA


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

    test = 1
