import numpy as np
import math
import sys
from common import logger

import matplotlib.pyplot as plt
from general import *


def select_samples(data, label, bands: int, thre: float, work_path: str):
    row, col = data.shape
    d_ndvi = reduce_feature_ndvi(data, 6, 1)
    ndvi_mean_path = "/home/tq/data_pool/zgq/semi_super/usa_data/crop_183_mean_ndvi.npy"
    crop_mean_curve = np.load(ndvi_mean_path)

    # get unrepeated labels
    l_list = []
    for i in label:
        if i not in l_list:
            l_list.append(i)
    print(l_list)
    n_vali = 0
    dout = np.ndarray([])
    dout_ndvi = np.ndarray([])

    # loop each label for selection
    for l in l_list:
        # print(l)
        d = keep_label(data, label, [l])
        dn = keep_label(d_ndvi, label, [l])
        dr, dc = d.shape
        dm = np.mean(d, axis=0)
        dm_ndvi = np.mean(dn, axis=0)
        idx = []
        if l == 6:  # other class
            pass
        else:  # crops
            for r in range(0, dr):
                d0 = dn[r, :]
                dd = np.fabs(d0 - dm_ndvi)
                ddmax = np.max(dd)
                if ddmax < thre:
                    n_vali += 1
                    idx.append(r)
                # print_progress_bar(r, row)

        d_select = d[idx, :]
        ds_row, ds_col = d_select.shape
        ds_label = np.ones((ds_row, 1))
        ds_label[:, :] = l
        d_select_l = np.hstack((d_select, ds_label))

        # ndvi
        dn_select = dn[idx, :]
        dn_select_l = np.hstack((dn_select, ds_label))
        print(l, ": new shape:", d_select_l.shape)
        if dout.ndim == 0:
            dout = d_select_l
            dout_ndvi = dn_select_l
        else:
            dout = np.vstack((dout, d_select_l))
            dout_ndvi = np.vstack((dout_ndvi, dn_select_l))

    print(" ")
    dout = data_shuffle(dout)
    dout, dout_label = strip_labels(dout)

    dout_ndvi = data_shuffle(dout_ndvi)
    dout_ndvi, dout_label_ndvi = strip_labels(dout_ndvi)
    print("new shape without other:", dout.shape, dout_ndvi.shape)

    # plt.figure(figsize=(8, 8))
    # for i in range(ds_row):
    #     plt.plot(dout_ndvi[i, :], color=plt.cm.Set1(dout_label[i]))
    # # plt.plot(dcrop_mean_ndvi, color="black")
    # my_x_ticks = np.arange(0, 13, 1)
    # my_y_ticks = np.arange(0, 1, 0.2)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    # plt.show()

    # deal with other now
    ##################################################
    dcrop_mean = np.mean(dout, axis=0)
    dother = keep_label(data, label, [6])
    dother_n = keep_label(d_ndvi, label, [6])
    dcrop_mean_ndvi = np.mean(dout_ndvi, axis=0)
    dr, dc = dother.shape
    drn, dcn = dother_n.shape

    idx = []
    idx2 = []
    for r in range(0, dr):
        d0 = dother_n[r, :]
        dd = np.fabs(d0 - dcrop_mean_ndvi)
        ddmax = np.max(dd)
        if ddmax < 0.15:
            idx.append(r)
        else:
            idx2.append(r)
        # print_progress_bar(r, row)
    d_select_other = dother[idx, :]
    ds_row, ds_col = d_select_other.shape
    ds_label = np.ones((ds_row, 1))
    ds_label[:, :] = 6
    d_select_o = np.hstack((d_select_other, ds_label))

    # other ndvi
    d_select_other_n = dother_n[idx, :]
    d_select_other_n_buyao = dother_n[idx2, :]
    d_select_o_n = np.hstack((d_select_other_n, ds_label))
    print("6: other:", d_select_o.shape)

    dout = np.hstack((dout, dout_label.reshape(dout_label.shape[0], -1)))
    dout_all = dout
    dout = np.vstack((dout, d_select_o))
    dout = data_shuffle(dout)
    dout, dout_label = strip_labels(dout)

    crop_mean_path = work_path + "crop_183_mean_ndvi_2.npy"
    np.save(crop_mean_path, dcrop_mean_ndvi)

    print("new shape with other:", dout.shape)

    plt.figure(figsize=(8, 8))
    for i in range(ds_row):
        plt.plot(d_select_other_n_buyao[i, :], color="red")
    plt.plot(dcrop_mean_ndvi, color="blue")
    my_x_ticks = np.arange(0, dcn, int(dcn / 10))
    my_y_ticks = np.arange(0, 1, 0.2)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.show()

    return dout, dout_label, dout_all, d_select_o


def select_tests(data, label, bands: int, thre: float, work_path: str):
    row, col = data.shape
    d_ndvi = reduce_feature_ndvi(data, 6, 1)
    ndvi_mean_path = "/home/tq/data_pool/zgq/semi_super/usa_data/crop_183_mean_ndvi.npy"
    crop_mean_curve = np.load(ndvi_mean_path)

    dout = data

    # deal with other now
    ##################################################
    # dother = data  # keep_label(data, label, [6])
    dother = data
    dother_n = d_ndvi  # keep_label(d_ndvi, label, [6])
    dcrop_mean_ndvi = crop_mean_curve
    dr, dc = dother.shape
    drn, dcn = dother_n.shape

    idx = []
    idx2 = []
    for r in range(0, dr):
        d0 = dother_n[r, :]

        dd = np.fabs(d0 - dcrop_mean_ndvi)
        dd1 = np.fabs(d0[0:-20] - dcrop_mean_ndvi[20:])
        dd2 = np.fabs(d0[20:] - dcrop_mean_ndvi[0:-20])
        ddmax = np.max(dd)
        dd1max = np.max(dd1)
        dd2max = np.max(dd2)
        if ddmax < 0.3 or dd1max < 0.3 or dd2max < 0.3:
            idx.append(r)
        else:
            idx2.append(r)
        """
        #######  method 2 
        """
        # new method
        flag = False
        n = int(len(d0) / 2)
        dmax = np.max(d0)
        dmin = np.min(d0)
        dbegin = np.mean(d0[:3])
        dend = np.mean(d0[-3:])
        dcenter = np.mean(d0[n - 1 : n + 1])
        # if (dmax - dmin) < 0.2:
        #     flag = True
        # if dcenter > dbegin and dcenter > dend:
        #     flag = False
        # else:
        #     flag = True

        # if flag:
        #     idx2.append(r)
        # else:
        #     idx.append(r)
        ##------------------------------------
        # if dbegin > 0.6 or dend > 0.6:
        #     flag = True
        # else:
        #     flag = False

        # if dcenter < 0.6:
        #     flag = True
        # else:
        #     flag = False

        # if flag:
        #     idx2.append(r)
        # else:
        #     idx.append(r)

        # print_progress_bar(r, row)
    d_select_test = dother[idx, :]
    d_select_test_label = label[idx].reshape(d_select_test.shape[0], -1)
    d_select_testall = np.hstack((d_select_test, d_select_test_label))

    d_select_buyao = dother_n[idx2, :]
    d_select_buyao_label = label[idx2]
    n_other = len(d_select_buyao_label[d_select_buyao_label == 6])
    print(n_other, d_select_buyao.shape[0])

    print("selected test:", d_select_testall.shape)

    plt.figure(figsize=(8, 8))
    for i in range(d_select_buyao.shape[0]):
        plt.plot(d_select_buyao[i, :], color=plt.cm.Set1(d_select_buyao_label[i]))
    # plt.plot(dcrop_mean_ndvi, color="blue")
    my_x_ticks = np.arange(0, dcn, int(dcn / 10))
    my_y_ticks = np.arange(0, 1, 0.2)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.show()

    return d_select_testall


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
    test_path = "/home/tq/zgq/zdata/crop-class/waterfall-pretrain/yunjie-10/mississipi/0401_0930_17_1_CoSoOtCoRi_L_REG_TEST_17.npz"
    # "/home/tq/zgq/zdata/crop-class/waterfall-pretrain/yunjie-10/mississipi/0401_0930_17_1_CoSoOtCoRi_L_REG_TRAIN_141516.npz"
    features_data, label_data = load_data(data_path)
    test_data, test_label = load_data(test_path)
    r = label_data.shape
    label_data = label_data.reshape(r[0], 1)
    stat_labels(label_data)

    all_data = np.hstack((features_data, label_data))

    all_data = data_shuffle(all_data)
    all_hard, all_soft = split_data(all_data, 0.05)

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
