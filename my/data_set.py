from datetime import datetime
from os import listdir
from pickle import load as pickle_load

import numpy as np
import tensorflow as tf
from keras.utils import timeseries_dataset_from_array as dataset_window
from matplotlib import pyplot as plt

from csv_process import SAVE_PATH
from hyperparameter import USE_EMD, USE_VMD, TIMESTEP, PRE_STEP, SPLIT_FRACTION, BATCH_SIZE, FEATURE_NUM


class Dataset:
    """
    自定义数据集类，提供了操作数据集的方法。
    读取、分割、制作样本
    """
    use_emd = USE_EMD
    use_vmd = USE_VMD
    timestep = TIMESTEP
    pre_step = PRE_STEP
    batch_size = BATCH_SIZE
    split_fraction = SPLIT_FRACTION
    feature_num = FEATURE_NUM

    def __init__(self, save_path=None):
        # 实例化之后是一个归一化好的原始数据集，没有分割和切片
        self.save_path = save_path or SAVE_PATH
        self.npy_list = listdir(self.save_path) + listdir(self.save_path + 'train_and_val')
        self.tpb = '{}_{}_{}'.format(self.timestep, self.pre_step, self.batch_size)  # 3者经常同时出现，便于引用
        self.load_data, self.raw_load_data, self.load_with_feature = self.load_girl_friend()
        self.data_length = len(self.load_data)
        self.all_data_scaler, self.load_scaler = self.get_scaler()  # 获取scaler实例
        self._train_end = int(self.split_fraction * self.data_length)  # 1728

    def load_girl_friend(self):
        """
        加载处理好的npy数据
        :return: load_data, raw_load_data, load_with_feature
        """
        # 加载负荷数据
        name1 = ['load_normalized.npy', 'raw_load.npy']
        for f in name1:
            if f not in self.npy_list:
                raise FileNotFoundError(f + ' 不存在')
        load_data, raw_load_data = [np.load(file=self.save_path + f) for f in name1]

        # 加载归一化的特征数据
        if self.use_emd:
            name2 = 'feature_{}_data_EMD.npy'.format(self.feature_num)
            if self.use_vmd:
                name2 = 'feature_{}_data_EMD_VMD.npy'.format(self.feature_num)
        else:
            name2 = 'feature_{}_data.npy'.format(self.feature_num)
        if name2 not in self.npy_list:
            raise FileNotFoundError('对应数据不存在')
        load_with_feature = np.load(file=self.save_path + name2)

        return load_data, raw_load_data, load_with_feature

    def plt_raw_load(self, time1, time2):
        """
        绘制原始负荷图
        :param time1: 起始日期，datetime
        :param time2: 结束日期，datetime
        :return: 图片窗口
        """
        # 读取原始负荷数据
        raw_load_data = self.raw_load_data

        # 转换为datetime64，查找索引中的位置
        if not all([isinstance(time1, datetime), isinstance(time2, datetime)]):
            raise TypeError('需要datetime格式')
        date_index_data = np.load(file=SAVE_PATH + 'date_index.npy')
        time1_np, time2_np = np.datetime64(time1), np.datetime64(time2)
        assert list(np.where(date_index_data == time1_np)[0]) != [], '日期有误'  # 如果没找到此日期
        assert list(np.where(date_index_data == time2_np)[0]) != [], '日期有误'  # 如果没找到此日期
        time1_index = int(np.where(date_index_data == time1_np)[0][0])
        time2_index = int(np.where(date_index_data == time2_np)[0][0])
        assert time1_index <= time2_index, '日期有误'

        # dt = date_index_data[time1_index:time2_index]
        dt_load = raw_load_data[time1_index:time2_index]
        plt.plot(dt_load)
        plt.ylabel('Load (MW)')
        plt.title("Load: ({})-({})".format(time1.strftime('%Y-%m-%d'), time2.strftime('%Y-%m-%d')), fontsize=20)
        plt.show()

    def get_scaler(self):
        """
        尝试读取csv_process阶段保存的scaler实例
        :return: scaler实例
        """
        name1 = 'feature_{}_scaler.sav'.format(self.feature_num)
        if name1 not in self.npy_list:
            raise FileNotFoundError(name1 + ' 不存在')
        if 'load_scaler.sav' not in self.npy_list:
            raise FileNotFoundError('load_scaler.sav 不存在')
        all_data_scaler = pickle_load(open(self.save_path + name1, 'rb'))
        load_scaler = pickle_load(open(self.save_path + 'load_scaler.sav', 'rb'))
        return all_data_scaler, load_scaler

    def _split(self):
        """
        分割训练集和验证集。（没有用到）
        :return: 训练集和验证集
        """
        train_end = int(self.split_fraction * self.data_length)
        train_data = self.load_data[:train_end]
        val_data = self.load_data[train_end]
        return train_data, val_data

    def get_train_x_y(self, col):
        """
        根据TIMESTEP，BATCH_SIZE，PRE_STEP组织好训练集的X和Y.
        :param col: 第几列。分解出来的负荷一列一列地预测
        :return: 已经切片并BATCH的train（内含X和Y） A tf.data.Dataset instance.
        """
        folder_name = 'feature_{}_clo_{}_load_train_{}'.format(self.feature_num, col, self.tpb)
        if folder_name not in self.npy_list:
            # 没有存好的就构建一个
            y_train = []
            for i in range(self._train_end - self.timestep - self.pre_step + 1):
                # 应该有 （长度-timestep-pre_step ）片
                y_train.append(self.load_with_feature[i + self.timestep:i + self.timestep + self.pre_step][:, col])
                # list:1704

            load_train = dataset_window(self.load_with_feature[:, col], np.array(y_train).reshape(-1, 1),
                                        sequence_length=self.timestep,
                                        batch_size=self.batch_size,
                                        end_index=self._train_end)
            # 1704 = 14*128+88    train_y最后一个对应于load_data第1728行（load_data[1727]）
            # Input shape: (128, 24, 1) (batch, timestep, feature)   Target shape: (128, 1)  (batch,pre_step)
            # 存储训练集
            load_train.save(self.save_path + 'train_and_val/' + folder_name)
            return load_train
        else:
            # 加载训练集
            return tf.data.Dataset.load(self.save_path + 'train_and_val/' + folder_name)

    def get_val_x_y(self, col):
        """
        把负荷和label变为数组，根据TIMESTEP，BATCH_SIZE，PRE_STEP组织好验证集的X和Y.
        :return: 已经切片并BATCH的valid（内含X和Y） A tf.data.Dataset instance.
        """
        folder_name = 'feature_{}_col_{}_load_val_{}'.format(self.feature_num, col, self.tpb)
        if folder_name not in self.npy_list:
            # 没有存好的就构建一个
            y_val = []
            for i in range(self._train_end, self.data_length - self.timestep - self.pre_step + 1):
                # 应该有 432-24-1+1 = 408个
                y_val.append(self.load_with_feature[i + self.timestep:i + self.timestep + self.pre_step][:, col])
                # list :408
            # 存储验证集的y，评估预测时会用到。
            np.save(self.save_path + 'train_and_val/' + 'feature_{}_col_{}_val_y_true_{}.npy'.format(self.feature_num,col, self.tpb),
                    np.array(y_val))
            # 验证集
            load_val = dataset_window(self.load_with_feature[self._train_end:][:, col], np.array(y_val).reshape(-1, 1),
                                      sequence_length=self.timestep, batch_size=self.batch_size)
            # Input shape: (128, 24, 1) (batch, timestep, feature)   Target shape:  (128, 1)  (batch,pre_step)
            # list(load_val)[0][0].numpy()  # debug对应无误
            # 存储验证集
            load_val.save(self.save_path + 'train_and_val/' + folder_name)
            return load_val
        else:
            # 加载验证集
            return tf.data.Dataset.load(self.save_path + 'train_and_val/' + folder_name)
