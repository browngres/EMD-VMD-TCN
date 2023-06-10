from os import listdir

import numpy as np
from PyEMD import EMD
from matplotlib import pyplot as plt


# 已有EMD分解过的npy文件的话，读取不用执行分解了
def check_emd_npy(save_path):
    """检查是否有imf的npy文件"""
    has_npy = False
    npy_list = listdir(save_path)
    for npy_str in npy_list:
        if npy_str == 'load_emd.npy':
            has_npy = True
            break
    return has_npy


def perform_emd(load: np.ndarray, save_path):
    """
    对负荷执行emd
    :param save_path: path to save npy
    :param load: 没有归一化的负荷 (None,1)
    :return: 分解后的 IMFs
    """
    load = load.flatten()
    length = load.shape[0]
    t = np.arange(0, length)
    emd = EMD()
    IMFs = emd.emd(load, t)
    # 存储为npy文件
    np.save(file=save_path + 'load_emd.npy', arr=IMFs)  # ndarray:(8,2160)
    return IMFs


def plot_emd(load, IMFs, divide=2):
    """
    画图展现EMD分解结果
    :param load: 未归一化的负荷
    :param IMFs:  IMFs
    :param divide: 查看多少数据：1表示全部。2表示前1/2，3表示前1/3。。。。。
    """
    num = IMFs.shape[0]
    length = IMFs.shape[1]
    load = load.flatten()
    t = np.arange(0, len(load))
    # 未分解
    plt.figure(figsize=(12, 10))  # 图片尺寸
    plt.subplot(num + 1, 1, 1)
    plt.plot(t[:int(length / divide)], load[:int(length / divide)], 'r', lw=1.1)
    plt.ylabel("Load (MW)")
    # 分解的
    for n in range(num):
        plt.subplot(num + 1, 1, n + 2)
        plt.plot(t[:int(length / divide)], IMFs[n][:int(length / divide)], 'g', lw=0.85)
        if n == num - 1:
            plt.ylabel("Res")  # 最后一个是残余分量
        else:
            plt.ylabel("IMF{}".format(n + 1))
    plt.title('Load EMD ', y=-1)
    plt.tight_layout()
    plt.show()
