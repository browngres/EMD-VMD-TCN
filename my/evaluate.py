import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from tcn import TCN
from tensorflow import keras

from data_set import Dataset
from hyperparameter import SAVED_MODEL_PATH, COL


def show_plot_every_single(plot_data, delta):
    """
    展示单点预测
    :param plot_data:三组数据：原始序列，实际未来点，预测未来点
    :param delta: 向后几个位置显示预测
    """
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]

    time_steps = list(range(-(plot_data[0].shape[0]), 0))  # [-6, -5, -4, -3, -2, -1]
    future = delta if delta else 0

    for i, val in enumerate(plot_data):
        if i:  # 第i>0个
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:  # 第i=0个
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.title("Single Step Prediction")
    plt.legend()
    plt.xlim([time_steps[0] - 2, 1])
    plt.xlabel("Time-Step")
    plt.show()


if __name__ == '__main__':
    # 加载数据集和模型
    d = Dataset()
    val_data = d.get_val_x_y(col=COL)
    model_file_name = SAVED_MODEL_PATH + "feature_{}_col_{}_best_model_{}.h5".format(d.feature_num, COL, d.tpb)
    model = keras.models.load_model(model_file_name, custom_objects={'TCN': TCN})
    # 真实值和预测值
    y_true = np.load(d.save_path + 'train_and_val/' + 'feature_{}_col_{}_val_y_true_{}.npy'.format(d.feature_num, COL, d.tpb)).reshape(-1, 1)
    y_pred = model.predict(val_data)
    # 画图
    length = len(y_pred)
    t = range(length)
    plt.figure(figsize=(14, 6))
    plt.plot(t, y_pred, 'D--', color='red', alpha=0.7, markersize=3, label='Prediction')
    plt.plot(t, y_true, '.-', c='purple', alpha=1, markersize=4, label='True')
    plt.xlim([-5, int(length * 1.05)])  # 拓展一下横轴范围，便于观看
    plt.xlabel("Time")
    plt.ylabel('Load')
    plt.legend()
    """
    # 一列一列就不用反归一化了
    y_pred = d.load_scaler.inverse_transform(y_pred).flatten()
    y_true = d.load_scaler.inverse_transform(y_true).flatten()  # debug过，对应csv无误
    """
    # 存下此次预测结果
    np.save(d.save_path + 'predicted/' + 'feature_{}_col_{}_pred_{}.npy'.format(d.feature_num, COL, d.tpb),y_pred)

    # 计算预测结果评价指标
    # RMSE
    print('RMSE: ', MSE(y_true, y_pred, squared=False))  # squared=False 代表使用RMSE
    # MAPE
    print('MAPE: ', MAPE(y_true, y_pred))
    # R2
    print('R2: ', R2(y_true, y_pred))

    plt.show()  # 显示图片放到计算评价指标后面