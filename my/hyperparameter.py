DEVICE = "cuda"  # 'cpu'  'cuda'
SAVED_MODEL_PATH = 'my/saved_model/'

SPLIT_FRACTION = 0.8  # 划分训练集测试集 8：2

# 只用EMD、用EMD+VMD、两者都不用
USE_EMD = True
USE_VMD = True
# 注意FEATURE_NUM一定要对，和上面两个要搭配。否则训练时会出问题。
FEATURE_NUM = 13  # 13:EMD+VMD 1:仅负荷  8:EMD   3:天气+负荷   16:EMD+VMD+天气+负荷
COL = 12  # 当前训练第几列（0~FEATURE_NUM-1）一共13列（0-12）。如果FEATURE_NUM = 1，COL就是0

EPOCHS = 200
TIMESTEP = 24  # 使用24个预测1个
PRE_STEP = 1
BATCH_SIZE = 128

# VMD
VMD_ALPHA = 1000  # alpha
VMD_K = 6  # K 分解模态（IMF）个数
VMD_TAU = 0  # tau 噪声容限，即允许重构后的信号与原始信号有差别。
VMD_DC = 0  # DC 若为0则让第一个IMF为直流分量/趋势向量
VMD_INIT = 1  # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
VMD_TOL = 1e-6  # 控制误差大小常量，决定精度与迭代次数
