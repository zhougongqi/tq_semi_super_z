# -*- coding: utf-8 -*-
import numpy as np
from common import logger
from general import *
from sklearn.cluster import KMeans, MiniBatchKMeans

import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    """
    label:  palm
    
    """
    n_bands = 11
    # data_path = "/home/tq/zgq/zdata/crop-class/waterfall-pretrain/yunjie-10/mississipi/0401_0930_17_1_CoSoOtCoRi_L_REG_TRAIN_141516.npz"
    # features_data, label_data = load_data(data_path)
    data_path = "/home/tq/data_pool/zgq/test_vortex/PE/td_all_label.npy"
    data = np.load(data_path)
    features_data, label_data = strip_labels(data)
    print(features_data.shape)
    print(label_data.shape)
    r = label_data.shape
    label_data = label_data.reshape(r[0], 1)
    stat_labels(label_data)

    all_data = np.hstack((features_data, label_data))

    all_data = data_shuffle(all_data)
    all_train, all_test = split_data(all_data, 0.1)
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

    # hard_label = hard_label.reshape(hard_label.shape[0], -1)
    # all_hard = np.hstack((hard_label, hard))
    # soft_label = np.hstack(
    #     (
    #         soft_label.reshape(soft_label.shape[0], -1),
    #         soft_label.reshape(soft_label.shape[0], -1),
    #     )
    # )

    # show tsne graph
    # ndvi_h = reduce_feature(hard, n_bands)

    # delete other class
    all_hard = np.hstack((hard, hard_label.reshape(hard_label.shape[0], -1)))
    all_hard = delete_label(all_hard, [5])
    hard, hard_label = strip_labels(all_hard)
    print(hard.shape)

    show_tsne.show_tsne(hard, hard_label)

    # save csv
    all_hard_list = all_hard.tolist()
    soft_list = soft.tolist()
    soft_label_list = soft_label.tolist()
    header = ["y"]
    header_soft = []
    header_soft_label = ["y", "y2"]
    for i in range(all_hard.shape[1] - 1):
        header.append("x_" + str(i + 1))
        header_soft.append("x_" + str(i + 1))

    with open(
        "/home/tq/zgq/zdata/crop-class/zgq/reduced/mississipi-allhard-reduced.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerows([header])
        writer.writerows(all_hard_list)
        f.close()

    with open(
        "/home/tq/zgq/zdata/crop-class/zgq/reduced/mississipi-soft-reduced.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerows([header_soft])
        writer.writerows(soft)
        f.close()

    with open(
        "/home/tq/zgq/zdata/crop-class/zgq/reduced/mississipi-soft-label-reduced.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerows([header_soft_label])
        writer.writerows(soft_label_list)
        f.close()
    # np.savetxt(
    #     "/home/tq/zgq/zdata/crop-class/zgq/mississipi-allhard.csv",
    #     all_hard,
    #     delimiter=",",
    # )
    # np.savetxt(
    #     "/home/tq/zgq/zdata/crop-class/zgq/mississipi-soft.csv", soft, delimiter=","
    # )
    # np.savetxt(
    #     "/home/tq/zgq/zdata/crop-class/zgq/mississipi-soft-label.csv",
    #     soft_label,
    #     delimiter=",",
    # )
    print("fin")

    # show_tsne.show_tsne(hard, hard_label)

    test = 1
