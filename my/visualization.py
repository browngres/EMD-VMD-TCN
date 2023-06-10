"""
一些数据可视化
"""

"""
# 查看指定日期的原始负荷数据
from datetime import datetime
from data_set import Dataset
d1 = Dataset()
time1 = datetime(2021, 1, 1)
time2 = datetime(2021, 3, 31,22)  #  (22点是当天最后一个)
d1.plt_raw_load(time1, time2)
# 2023-01-13 到 2023-04-18  95天, 2280小时
"""

"""
# 查看EMD图像
import numpy as np
from csv_process import SAVE_PATH
from load_EMD import plot_emd, check_emd_npy, perform_emd
raw_load = np.load(file=SAVE_PATH + 'raw_load.npy')
IMFs = np.load(file=SAVE_PATH + 'load_emd.npy') if check_emd_npy(SAVE_PATH) else perform_emd(raw_load, SAVE_PATH)
plot_emd(raw_load, IMFs, divide=2)  # divide代表前n分之一
"""

"""
# 查看VMD图像
import numpy as np
from csv_process import SAVE_PATH
from hyperparameter import VMD_K, VMD_ALPHA
from load_secondary_VMD import check_sec_vmd_npy, perform_vmd, plot_vmd
IMFs = np.load(file=SAVE_PATH + 'load_emd.npy')
IMF1 = IMFs[0]
# s_vmd = np.load(file=SAVE_PATH + 'load_s_vmd_a_{}_K_{}.npy'.format(VMD_ALPHA, VMD_K)) \
#     if check_sec_vmd_npy(SAVE_PATH) else perform_vmd(IMF1, SAVE_PATH)
s_vmd = perform_vmd(IMF1, SAVE_PATH)
plot_vmd(s_vmd, divide=2)  # divide代表前n分之一
"""

"""
# 参数对比 A
import matplotlib.pyplot as plt

x = [12, 24, 36, 48]
ks_2_rmse = [935.71, 797.10, 676.58, 599.89]
ks_2_mape = [2.42, 2.28, 1.82, 1.67]
ks_3_rmse = [1029.40, 710.74, 618.85, 581.78]
ks_3_mape = [2.69, 1.85, 1.63, 1.56]
# 大概是380倍
plt.figure(figsize=(8, 4))

plt.plot(x, ks_2_rmse, '.:g', lw=1.5, label='kernal_size_2_RMSE')
plt.plot(x, ks_3_rmse, '.-b', lw=1.5, label='kernal_size_3_RMSE')
plt.xticks(x)
plt.ylim((400, 1200))
plt.ylabel('RMSE (MW)')
plt.xlabel('kernel_size')
plt.legend(loc='upper left')

plt.twinx()
plt.plot(x, ks_2_mape, '1g', ms=8, label='kernal_size_2_MAPE')
plt.plot(x, ks_3_mape, '1b', ms=8, label='kernal_size_3_MAPE')
plt.xticks(x)
plt.ylabel('MAPE %')
plt.ylim((1, 3.2))
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
"""


# 5个模型结果对比
import matplotlib.pyplot as plt
import numpy as np
from hyperparameter import TIMESTEP, PRE_STEP, BATCH_SIZE, SPLIT_FRACTION
import pickle

from csv_process import SAVE_PATH
tpb = '{}_{}_{}'.format(TIMESTEP, PRE_STEP, BATCH_SIZE)
load_scaler = pickle.load(open(SAVE_PATH + 'load_scaler.sav', 'rb'))
# TRUE
true = np.load(SAVE_PATH+'load_normalized.npy').reshape(-1,1)
true = true[int(SPLIT_FRACTION*true.shape[0])+TIMESTEP:]
true = load_scaler.inverse_transform(true).flatten()
# TCN
pred_1 = np.load(SAVE_PATH+'predicted/' + 'feature_1_pred_{}.sav'.format(tpb), allow_pickle=True)
# EMD-VMD-TCN
pred_2 = np.load(SAVE_PATH+'predicted/' + 'feature_13_col_all_pred_{}.npy'.format(tpb))
#LSTM
pred_3 = np.load(SAVE_PATH+'predicted/' + 'LSTM_300_y_pred_{}.npy'.format(tpb))
pred_3 = load_scaler.inverse_transform(pred_3).flatten()
# CNN-BiLSTM
pred_4 = np.load(SAVE_PATH+'predicted/' + 'y_predict_cnn_bilstm.npy')
pred_4 = load_scaler.inverse_transform(pred_4).flatten()
# EMD-TCN
pred_5 = np.load(SAVE_PATH+'predicted/' + 'feature_8_col_all_pred_{}.npy'.format(tpb))

length = len(true)
t = range(length)
plt.figure(figsize=(16,8))
plt.plot(t, true, '-', color='purple', alpha=1, label='True',lw=2)
plt.plot(t, pred_1, 'D--', c='blue', alpha=0.5, markersize=3, label='TCN')
plt.plot(t, pred_2, 'v:', c='red', alpha=0.5, markersize=4, label='EMD-VMD-TCN')
plt.plot(t, pred_3, '+--', c='green', alpha=0.5, markersize=4, label='LSTM')
plt.plot(t, pred_4, 'x--', c='gray', alpha=0.5, markersize=4, label='CNN-BiLSTM')
plt.plot(t, pred_5, '2--', c='pink', alpha=0.8, markersize=4, label='EMD-TCN')
plt.xlim([-5, int(length * 1.05)])  # 拓展一下横轴范围，便于观看
plt.xlabel("Time")
plt.ylabel("Load/MW")
plt.legend()
plt.show()
