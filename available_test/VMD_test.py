# https://blog.csdn.net/weixin_56243568/article/details/124364642
import matplotlib.pyplot as plt
import numpy as np
from vmdpy import VMD

# 测试信号及其参数
T = 1000
fs = 1 / T
t = np.arange(1, T + 1) / T
f_1 = 2
f_2 = 24
f_3 = 288
v_1 = (np.cos(2 * np.pi * f_1 * t))
v_2 = 1 / 4 * (np.cos(2 * np.pi * f_2 * t))
v_3 = 1 / 16 * (np.cos(2 * np.pi * f_3 * t))
v = [v_1, v_2, v_3]  # 测试信号所包含的各成分
f = v_1 + v_2 + v_3 + 0.1 * np.random.randn(v_1.size)  # 测试信号

alpha = 2000  # alpha 带宽限制经验取值为抽样点长度1.5-2.0倍
tau = 0  # tau 噪声容限，即允许重构后的信号与原始信号有差别。
K = 3  # K 分解模态（IMF）个数
DC = 0  # DC 若为0则让第一个IMF为直流分量/趋势向量
init = 1  # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
tol = 1e-7  # 控制误差大小常量，决定精度与迭代次数
u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)  # 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
# 画原始信号和它的各成分
plt.figure(figsize=(10, 7))
plt.subplot(K + 1, 1, 1)
plt.plot(t, f)
for i, y in enumerate(v):
    plt.subplot(K + 1, 1, i + 2)
    plt.plot(t, y)
plt.suptitle('Original input signal and its components')
plt.show()

# 分解出来的各IMF分量
plt.figure(figsize=(10, 7), dpi=80)
for i in range(K):
    plt.subplot(K, 1, i + 1)
    plt.plot(t, u[i, :])
    plt.title('IMF{}'.format(i + 1))
plt.show()
# 分解出来的各IMF分量的频谱
print(u_hat.shape, t.shape, omega.shape)
plt.figure(figsize=(10, 7), dpi=80)
for i in range(K):
    plt.subplot(K, 1, i + 1)
    plt.plot(t, u_hat[:, i])
    plt.title('spectra of the modes{}'.format(i + 1))
# plt.tight_layout()
plt.show()
# 各IMF的中心频率
plt.figure(figsize=(12, 7), dpi=80)
for i in range(K):
    plt.subplot(K, 1, i + 1)
    plt.plot(omega[:, i])  # X轴为迭代次数，y轴为中心频率
    plt.title('mode center-frequencies{}'.format(i + 1))
plt.tight_layout()
plt.show()
