import numpy as np
from matplotlib import pyplot as plt
from vmdpy import VMD

t = np.linspace(0, 1, 200)
s = 10*(np.cos(2 * np.pi * 2 * t)) # + 20 * (np.cos(2 * np.pi * 4 * t))
ran = np.random.random(size=200)
s += ran
# s = 10*(np.cos(2 * np.pi * 2 * t)) + 2*(np.cos(2 * np.pi * 100 * t))

alpha = 400  # alpha 带宽限制经验取值为抽样点长度1.5-2.0倍
tau = 0  # tau 噪声容限，即允许重构后的信号与原始信号有差别。
K = 2  # K 分解模态（IMF）个数
DC = 0  # DC 若为0则让第一个IMF为直流分量/趋势向量
init = 1  # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
tol = 1e-7  # 控制误差大小常量，决定精度与迭代次数
u, u_hat, omega = VMD(s, alpha, tau, K, DC, init, tol)

plt.figure(figsize=(12,12))
plt.subplot(K + 2, 1, 1)
plt.plot(t, s, 'r',label="Input signal")
plt.xlabel("Time [s]")
plt.legend()
plt.subplot(K + 2, 1, 2)
plt.plot(t,ran,'b',label='noise')
plt.legend()

for i in range(K):
    plt.subplot(K +2, 1, i + 3)
    plt.plot(t, u[i, :], 'g')
    plt.xlabel("Time [s]")
    plt.title('IMF{}'.format(i + 1))
plt.tight_layout()
plt.show()
