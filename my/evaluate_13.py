"""
计算13个预测结果最终RMSE（反归一化的）
"""
from pickle import load

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2

from csv_process import SAVE_PATH
from hyperparameter import FEATURE_NUM, TIMESTEP, PRE_STEP, BATCH_SIZE

# 读取13列预测和真实值 （408，13）
pred_list, true_list = [], []
tpb = '{}_{}_{}'.format(TIMESTEP, PRE_STEP, BATCH_SIZE)
pre_name = ['feature_{}_col_{}_pred_{}.npy'.format(FEATURE_NUM, i, tpb) for i in range(FEATURE_NUM)]
true_name = ['feature_{}_col_{}_val_y_true_{}.npy'.format(FEATURE_NUM,i, tpb) for i in range(FEATURE_NUM)]
for i in range(FEATURE_NUM):
    pre = np.load(SAVE_PATH + 'predicted/' + pre_name[i])
    pred_list.append(pre.flatten())
    tru = np.load(SAVE_PATH + 'train_and_val/' + true_name[i])
    true_list.append(tru.flatten())

# 读取归一化器
all_scaler = load(open(SAVE_PATH + 'feature_{}_scaler.sav'.format(FEATURE_NUM), 'rb'))
# 预测值
y_pred = np.array(pred_list).T
y_pred = all_scaler.inverse_transform(y_pred)
y_pred_sum = y_pred.sum(axis=1)
np.save(SAVE_PATH + 'predicted/'+'feature_{}_col_all_pred_{}.npy'.format(FEATURE_NUM, tpb), y_pred_sum)
# 真实值
y_true = np.array(true_list).T
y_true = all_scaler.inverse_transform(y_true)
y_true_sum = y_true.sum(axis=1)
np.save(SAVE_PATH + 'train_and_val/'+'feature_{}_col_all_true_{}.npy'.format(FEATURE_NUM, tpb), y_true_sum)
# 评价
print('RMSE: ', MSE(y_true_sum, y_pred_sum, squared=False))  # squared=False 代表使用RMSE
print('MAPE: ', MAPE(y_true_sum, y_pred_sum))
print('R2: ', R2(y_true_sum, y_pred_sum))
# 画图
length = y_pred.shape[0]
plt.figure(figsize=(12, 6))
plt.plot(range(length), y_pred_sum, 'D--', color='red', alpha=0.7, markersize=3, label='Prediction')
plt.plot(range(length), y_true_sum, '.-', color='purple', alpha=1, markersize=4, label='History')
plt.ylabel("Load (MW)")
plt.title('EMD-VMD-TCN')
plt.legend()
plt.show()
