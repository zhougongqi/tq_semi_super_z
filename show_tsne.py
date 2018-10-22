# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from common import logger
from general import *
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA

import show_tsne
import csv
import pandas


def load_data(data_path: str):
    try:
        features_data = np.load(data_path)["features"]
        label_data = np.load(data_path)["labels"]
        print(features_data.shape)
        print(label_data.shape)
    except Exception as e:
        logger.error("load ori-data failed!".center(40, "!"))
        return None

    return features_data, label_data


def show_tsne(data, label):

    X, y = data, label
    n_samples, n_features = X.shape

    """t-SNE"""
    tsne = manifold.TSNE(
        n_components=2,
        init="pca",
        random_state=501,
        n_iter=5000000,
        verbose=1,
        perplexity=25,
        learning_rate=500,
        early_exaggeration=20,
    )
    X_tsne = tsne.fit_transform(X)

    print(
        "Org data dimension is {}. Embedded data dimension is {}".format(
            X.shape[-1], X_tsne.shape[-1]
        )
    )

    """嵌入空间可视化"""
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(
            X_norm[i, 0],
            X_norm[i, 1],
            str(y[i]),
            color=plt.cm.Set1(y[i]),
            fontdict={"weight": "bold", "size": 9},
        )
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    n_bands = 6
    data_path = "/home/tq/zgq/zdata/crop-class/waterfall-pretrain/yunjie-10/mississipi/0401_0930_17_1_CoSoOtCoRi_L_REG_TRAIN_141516.npz"
    features_data, label_data = load_data(data_path)
    r = label_data.shape
    label_data = label_data.reshape(r[0], 1)
    stat_labels(label_data)

    all_data = np.hstack((features_data, label_data))

    all_data = data_shuffle(all_data)
    all_train, all_test = split_data(all_data, 0.2)
    all_hard, all_soft = split_data(all_train, 0.01)

    # get hard, soft, test data
    print("all_hard", all_hard.shape)
    print("all_soft", all_soft.shape)
    print("all_test", all_test.shape)

    # strip label from all array
    hard, hard_label = strip_labels(all_hard)
    soft, soft_label = strip_labels(all_soft)
    test, test_label = strip_labels(all_test)

    stat_labels(hard_label)

    # reduce time resolution by 15 times
    hard = reduce_feature_ndvi(hard, n_bands)
    soft = reduce_feature_ndvi(soft, n_bands)

    # show tsne graph
    # ndvi_h = reduce_feature(hard, n_bands)

    # # delete other class
    # all_hard = np.hstack((hard, hard_label.reshape(hard_label.shape[0], -1)))
    # all_hard = delete_label(all_hard, [1, 3, 6])
    # hard, hard_label_1 = strip_labels(all_hard)
    # print(hard.shape)
    hard_label_1 = hard_label

    pca = PCA(n_components=2)
    pca.fit(hard)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    hard_new = pca.transform(hard)

    for l in range(len(hard_label_1)):
        if hard_label_1[l] == 0:
            plt.scatter(hard_new[l, 0], hard_new[l, 1], color="red", marker=".")
        if hard_label_1[l] == 1:
            plt.scatter(hard_new[l, 0], hard_new[l, 1], color="blue", marker=".")
        if hard_label_1[l] == 2:
            plt.scatter(hard_new[l, 0], hard_new[l, 1], color="green", marker=".")
        if hard_label_1[l] == 3:
            plt.scatter(hard_new[l, 0], hard_new[l, 1], color="gold", marker=".")
        if hard_label_1[l] == 3:
            plt.scatter(hard_new[l, 0], hard_new[l, 1], color="gray", marker=".")
    plt.show()

    # for h in range(hard.shape[0]):
    #     plt.plot(hard[h, :], color=plt.cm.Set1(hard_label_1[h]))
    # plt.show()

    show_tsne(hard, hard_label_1)

    test = 1
