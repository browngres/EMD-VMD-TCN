import pandas as pd
from tensorflow import keras

from keras_example_weather_forecasting.hyperparamter import split_fraction, past, future, step, batch_size

SAVE_PATH = '../../data/Jena Climate dataset/'
csv_name = "jena_climate_2016_selected.csv"
df = pd.read_csv(SAVE_PATH + csv_name)

train_split = int(split_fraction * int(df.shape[0]))  # 37365


# 归一化
def normalize(data, train_split):
    """z—score标准化: 将原始数据集归一化为均值为0、方差1的数据集
    归一化方式要求原始数据的分布可以近似为正态分布"""

    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


features = df.iloc[:, 1:]
df_nor = normalize(features.values, train_split)
df_nor = pd.DataFrame(df_nor)

# 分割
train_data = df_nor.loc[0: train_split - 1]  # 前train_split行
val_data = df_nor.loc[train_split:]

# 训练集
start = past + future  # 720+72   #学习120小时，预测12小时后的1小时
end = start + train_split  # 720+72+37365

x_train = train_data[[i for i in range(7)]].values
y_train = df_nor.iloc[start:end][[1]]  # 温度  72次观察后的温度将是用作标签。
# The training dataset labels starts from the 792nd observation (720 + 72).
sequence_length = int(past / step)  # 720/6   120h

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train, y_train, sequence_length=sequence_length, sampling_rate=step, batch_size=batch_size, )


# 验证集
# The validation label dataset must start from 792 after train_split, hence we must add past + future to label_start.
x_end = len(val_data) - past - future
label_start = train_split + past + future

x_val = val_data.iloc[:x_end][[i for i in range(7)]].values
y_val = df_nor.iloc[label_start:][[1]]
dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val, y_val, sequence_length=sequence_length, sampling_rate=step, batch_size=batch_size, )

for batch in dataset_train.take(1):
    inputs, targets = batch
    print("Input shape:", inputs.numpy().shape)  # (256, 120, 7)
    print("Target shape:", targets.numpy().shape)  # (256, 1)
