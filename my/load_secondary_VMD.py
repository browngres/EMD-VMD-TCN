"""
读取负荷EMD分解后的结果，对其中的IMF1再次进行VMD
"""
from os import listdir

import numpy as np
from matplotlib import pyplot as plt
from vmdpy import VMD

from hyperparameter import VMD_K, VMD_DC, VMD_TAU, VMD_TOL, VMD_INIT, VMD_ALPHA
from load_EMD import check_emd_npy


# 已有VMD分解过的npy文件的话，读取不用执行分解了
def check_sec_vmd_npy(save_path):
    """检查是否有VMD的npy文件"""
    has_npy = False
    npy_list = listdir(save_path)
    for npy_str in npy_list:
        if npy_str == 'load_s_vmd_a_{}_K_{}.npy'.format(VMD_ALPHA, VMD_K):
            has_npy = True
            break
    return has_npy


def perform_vmd(imf1: np.ndarray, save_path):
    """
    对EMD的IMF1执行VMD
    :param imf1: 要分解的IMF1（来自EMD结果）
    :param save_path: path to save npy
    :return: VMD分解后的 IMFs
    """
    length = imf1.shape[0]
    t = np.arange(0, length)
    # 调用vmd，注意参数顺序。
    s_vmd, spectra, omega = VMD(imf1, VMD_ALPHA, VMD_TAU, VMD_K, VMD_DC, VMD_INIT, VMD_TOL)
    # 下断点查看频率  omega[-1]
    # 存储为npy文件
    np.save(file=save_path + 'load_s_vmd_a_{}_K_{}.npy'.format(VMD_ALPHA, VMD_K), arr=s_vmd)
    return s_vmd


def plot_vmd(s_vmd, divide=2):
    """
    画图展现VMD分解结果
    :param s_vmd: VMD分解结果
    :param divide: 查看多少数据：1表示全部。2表示前1/2，3表示前1/3。。。。。
    """
    # 加载负荷的EMD的IMF1数据
    from csv_process import SAVE_PATH
    if not check_emd_npy(SAVE_PATH):
        raise FileNotFoundError('load_emd.npy 不存在')
    emd_imf = np.load(file=SAVE_PATH + 'load_emd.npy')  # ndarray:(8,2172)
    imf1 = emd_imf[0]
    # 准备画图
    length = s_vmd.shape[1]
    num = s_vmd.shape[0]  # 和K是相等的
    t = np.arange(0, len(imf1))  # len(imf1) 和length也是相等的
    # 未分解
    plt.figure(figsize=(12, 12))  # 图片尺寸
    plt.subplot(num + 1, 1, 1)
    plt.plot(t[:int(length / divide)], imf1[:int(length / divide)], 'r', lw=1.1)
    plt.ylabel("EMD-IMF1")
    # 分解的
    for n in range(num):
        plt.subplot(num + 1, 1, n + 2)
        plt.plot(t[:int(length / divide)], s_vmd[n][:int(length / divide)], 'g', lw=0.85)
        plt.ylabel("Mode {}".format(n + 1))
    plt.suptitle('VMD results')
    plt.tight_layout()
    plt.show()
