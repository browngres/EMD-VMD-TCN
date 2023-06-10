import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_xy_kfolds(split_index=None, timesteps=1000):
    """
    load exchange rate dataset and preprocess it, then split it into k-folds for CV
    :param split_index: list, the ratio of whole dataset as train set
    :param timesteps: length of a single train x sample
    :return: list, [train_x_set,train_y_set,test_x_single,test_y_single]
    """
    if split_index is None:
        split_index = [0.5, 0.6, 0.7, 0.8, 0.9]
    df = pd.read_csv('exchange_rate.csv',header=None)
    n = len(df)
    folds = []
    enc = MinMaxScaler()
    df = enc.fit_transform(df)
    for split_point in split_index:
        train_end = int(split_point * n)
        train_x, train_y = [], []
        for i in range(train_end - timesteps):
            train_x.append(df[i:i + timesteps])
            train_y.append(df[i + timesteps])
        train_x = np.array(train_x)  # 每次看1000个
        train_y = np.array(train_y)  # 用1000个预测下一个
        test_x = df[train_end - timesteps + 1:train_end + 1]  # 给你没见过的1000个
        test_y = df[train_end + 1]  # 这是没见过的1000个的真实答案。
        folds.append((train_x, train_y, test_x, test_y))
    return folds, enc


if __name__ == '__main__':
    get_xy_kfolds()
