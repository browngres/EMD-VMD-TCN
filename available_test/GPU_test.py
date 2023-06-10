import timeit  # 性能分析模块

import tensorflow as tf

# https://blog.csdn.net/wyyyyyyyy_/article/details/124490343

# 查看版本
print(tf.__version__)
# CUDA是否可用
print(tf.test.is_built_with_cuda())
# GPU是否可用
print(tf.config.list_physical_devices())

# 创建在cpu环境上运算的两个矩阵
# tf.device：指定模型运行的具体设备
with tf.device('/cpu:0'):
    # 用于从“服从指定正态分布的序列”中随机取出指定个数的值
    # shape：输出张量的形状，必选
    # mean：正态分布的均值，默认为0
    # stddev：正态分布的标准差，默认为1.0
    # dtype：输出的类型，默认为tf.float32
    # seed：随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
    cpu_a = tf.random.normal(shape=[10000, 1000],seed=666)
    cpu_b = tf.random.normal([1000, 2000],seed=666)
    print(cpu_a.device, cpu_b.device)

# 创建在gpu环境上运算的两个矩阵
with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([10000, 1000],seed=666)
    gpu_b = tf.random.normal([1000, 2000],seed=666)
    print(gpu_a.device, gpu_b.device)


# cpu矩阵相乘
def cpu_run():
    with tf.device('/cpu:0'):
        # 将矩阵a乘以矩阵b，生成a*b
        c = tf.matmul(cpu_a, cpu_b)
    return c


# gpu矩阵相乘
def gpu_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(gpu_a, gpu_b)
    return c


# 正式计算前需要热身，避免将初始化时间结算在内
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('warm up:', cpu_time, gpu_time)

# 正式计算
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('run time:', cpu_time, gpu_time)

