# -*- coding: UTF-8 -*-
from pickle import dump

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from hyperparameter import USE_EMD,USE_VMD, VMD_ALPHA, VMD_K
from load_EMD import check_emd_npy, perform_emd
from load_secondary_VMD import perform_vmd, check_sec_vmd_npy

DATA_PATH = "data/巴塞罗那/"
SAVE_PATH = "data/processed/"

if __name__ == "__main__":
    # 读csv文件 日期识别、设置为索引
    file_name = r'Total Load - Day Ahead_20210101-20210401.csv'
    pd_csv = pd.read_csv(DATA_PATH + file_name, parse_dates=['Time (CET/CEST)'], index_col=['Time (CET/CEST)'])
    load = pd_csv.iloc[:, 1:]  # 第2列是真实负荷
    # 拼成一张表
    data = pd.concat([load], axis=1)  # 既有data又有load是因为以前用了其他特征，比如天气
    # 缺失值用前面一行填充
    data.fillna(axis=0, method='ffill', inplace=True)
    # 转换为array
    load_np = np.array(data.iloc[:, 0])  # 一维数组 (2160,)

    if USE_EMD:
        # 尝试读取文件，没有就进行分解。
        # 进行EMD分解
        IMFs = np.load(file=SAVE_PATH + 'load_emd.npy') if check_emd_npy(SAVE_PATH) else perform_emd(load_np, SAVE_PATH)
        # ndarray:(8,2160)  7*IMF+Res
        data_np = np.array(IMFs.T)
        if USE_VMD:
            IMF1 = IMFs[0]  # 取第一行  (2160,)
            # 对IMF1进行VMD
            s_vmd = np.load(file=SAVE_PATH + 'load_s_vmd_a_{}_K_{}.npy'.format(VMD_ALPHA, VMD_K)) \
                if check_sec_vmd_npy(SAVE_PATH) else perform_vmd(IMF1, SAVE_PATH)
            # (VMD_K,2160)
            # EMD除第一列的7列,VMD的6列,添加进data，负荷不要了
            data_np = np.concatenate((IMFs[1:].T, s_vmd.T), axis=1)  # (2160,13)
    # 归一化
    all_data_scaler = MinMaxScaler()
    data_normalized = all_data_scaler.fit_transform(data_np)
    load_scaler = MinMaxScaler()  # 如果不用模态分解，data中的负荷已经归一化了，下面是为了有一个单独对负荷的归一化器
    load_normalized = load_scaler.fit_transform(load_np.reshape(-1, 1))  # 只针对负荷的归一化器(预测完反归一化会用到)
    length = data_normalized.shape[0]  # 2160
    feature_num = data_np.shape[1]  # (13列 EMD除第一列:7 + VMD:6)   (8列 只用EMD)  （1列，不用模态分解）
    # 保存文件
    try:
        np.save(SAVE_PATH + 'date_index.npy', data.index)  # 日期索引
        np.save(SAVE_PATH + 'raw_load.npy', load_np)  # 未归一化的负荷
        np.save(SAVE_PATH + 'load_normalized.npy', load_normalized)  # 归一化后的负荷

        # 保存特征（没有切）
        if USE_EMD and USE_VMD:  # 使用EMD+VMD
            np.save(SAVE_PATH + 'feature_{}_data_EMD_VMD.npy'.format(feature_num), data_normalized)
        if USE_EMD and (not USE_VMD):  # 仅使用EMD
            np.save(SAVE_PATH + 'feature_{}_data_EMD.npy'.format(feature_num), data_normalized)
        else:  # 不使用模态分解
            np.save(SAVE_PATH + 'feature_{}_data.npy'.format(feature_num), data_normalized)
        # 保存带参数的MinMaxScaler备用
        dump(all_data_scaler, open(SAVE_PATH + 'feature_{}_scaler.sav'.format(feature_num), 'wb'))
        dump(load_scaler, open(SAVE_PATH + 'load_scaler.sav', 'wb'))

        print("CSV数据处理成功 OK")
    except FileNotFoundError:
        print("CSV保存失败 ERROR")
