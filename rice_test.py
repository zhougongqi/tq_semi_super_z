import numpy as np
import math
import sys
from common import logger

import matplotlib.pyplot as plt
from general import *


if __name__ == "__main__":
    """
    label:  mississipi
    #     "Corn": 0,
    #     "Soybeans": 1,
    #     "Cotton": 2,
    #     "Rice": 3,
    #     "Other": 6
    
    """
    n_bands = 6
    work_path = "/home/tq/data_pool/zgq/semi_super/usa_data/"
    data_path = "/home/tq/zgq/zdata/crop-class/waterfall-pretrain/yunjie-10/mississipi/0401_0930_17_1_CoSoOtCoRi_L_REG_TRAIN_141516.npz"
    # "/home/tq/zgq/zdata/crop-class/waterfall-pretrain/yunjie-10/mississipi/0401_0930_17_1_CoSoOtCoRi_L_REG_TEST_17.npz"
    # "/home/tq/zgq/zdata/crop-class/waterfall-pretrain/yunjie-10/mississipi/0401_0930_17_1_CoSoOtCoRi_L_REG_TRAIN_141516.npz"
    features_data, label_data = load_data(data_path)
    r = label_data.shape
    label_data = label_data.reshape(r[0], 1)
    stat_labels(label_data)

    all_data = np.hstack((features_data, label_data))

    all_data = data_shuffle(all_data)
    all_train, all_test = split_data(all_data, 0.1)
    all_hard, all_soft = split_data(all_train, 0.1)

    # get hard, soft, test data
    print("all_hard", all_hard.shape)
    print("all_soft", all_soft.shape)
    print("all_test", all_test.shape)

    # strip label from all array
    hard, hard_label = strip_labels(all_hard)
    soft, soft_label = strip_labels(all_soft)
    test, test_label = strip_labels(all_test)

    stat_labels(hard_label)

    # # reduce time resolution by 15 times
    # hard = reduce_feature(hard, n_bands)
    # soft = reduce_feature(soft, n_bands)
    print("reduced hard shape: ", hard.shape)

    all_hard = np.hstack((hard, hard_label.reshape(hard_label.shape[0], -1)))

    # select valid hard data with right temporal shape
    s_hard, s_hard_label, s_hard_all, s_other_all = select_samples(
        hard, hard_label, 6, 0.2, work_path
    )
    # s_test_all = select_tests(hard, hard_label, 6, 0.2, work_path)

    outpath = work_path + "xxx.npy"
    np.save(outpath, s_test_all)

    # outpath = work_path + "selected_traindata_other_f78_per2.npy"
    # np.save(outpath, s_other_all)

    # outpath = work_path + "selected_traindata_all_f78_per2.npy"
    # np.save(outpath, s_hard_all)

    test = 1
