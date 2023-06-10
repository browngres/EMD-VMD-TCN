# 使用方法

1. Pycharm 打开项目，右键`my`设置为`Source Root`，右键`reference`设置为`Excluded`。
2. `environment`文件夹下有环境依赖（不做*环境闭眼迷糊人*或者靠*conda*）
    - 法一（推荐）pipdeptree.txt
        - （参考依赖树文件，只需要按顺序手动安装一级的 5 个包即可，只有 tensorflow 要求版本一致，2.11 不支持 GPU）
        - (由于后来装的 keras-tcn 要求有 tensorflow，防止 keras-tcn 变成一级，在依赖树中手动调整了它的级别)
        - tensorflow==2.10.0
        - scikit-learn==1.2.2
        - pandas==1.5.3
        - matplotlib==3.7.1
        - EMD-signal==1.4.0
        - vmdpy==0.2 _可选，提供有单文件用于 import_
        - keras-tcn==3.5.0 _可选，提供有单文件用于 import_
        - pip-autoremove==0.10.0 可选
        - pipdeptree==2.5.2 可选
    - 法二（不推荐）requirement.txt
        - `pip install -r requirement.txt`
3. `available_test`为环境可用性测试，自行根据名称使用
4. （数据集 csv，npy 均已经设置为 gitignore）(所用数据集仅有时间和负荷两列)
5. 运行。(要用到的 py 文件均在`my`文件夹下)

    - **流程：修改参数`hyperparameter.py`---->运行`csv_process.py`---->运行`training.py`
      ---->运行`evaluate.py`运行---->`evaluate_13.py`**
    - 参数说明：
        - `USE_EMD`、`USE_VMD`、`FEATURE_NUM`一定要设置对，因为所有处理数据都会保存下来，以便重复运行。
        - `COL`用于控制当前是第几列（0-12），每一列都要运行`training.py`和`evaluate.py`。 依次更改`COL`到 12，运行完后才能运行`evaluate_13.py`，否则叠加结果没有意义。
        - `hyperparameter`更改了 epoch、timestep、pre_step 任意一个之后，必须先运行`training.py`，再运行`evaluate.py`，。
    - `csv_process.py`只需运行一次即可（只要`USE_EMD`、`USE_VMD`、`VMD_ALPHA`、`VMD_K` 没有变动）。
    - `training.py`和`evaluate.py`不可同时运行。前者会不断更新保存`best_model`，后者则是根据`hyperparameter.py`调用`best_model`。
    - `visualization.py`可视化：
        - 绘制指定日期的原始负荷数据
        - 绘制 EMD/VMD 图像
        - 参数对比
        - 5 个模型结果对比

6. `reference`是参考代码和示例。用于学习完整流程（数据集本地化、代码重构、复现）
    - `From_Keras` 来自 Keras 文档示例
        - [engine noise classification](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/)
            > CNN、时间序列分类。
        - [timeseries_weather_forecasting](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)
            > LSTM、时间序列预测。 `timeseries_weather_forecasting.py`是原文件。
            > 对此示例进行解耦重构，在`my/keras_example_weather_forecasting`文件夹下。
    - `From_Keras_tcn` [来自 keras-tcn 包的示例](https://github.com/philipperemy/keras-tcn)
        - [exchange_rate](https://github.com/philipperemy/keras-tcn/tree/master/tasks/exchange_rate)
            > TCN、时间序列预测。 这个有 ipynb
        - [milk_production](https://github.com/philipperemy/keras-tcn/blob/master/tasks/time_series_forecasting.py)
            > TCN 、时间序列预测。 极简示例。 (It's a very naive (toy) example to show how to do time series forecasting.)
    - 另外放了两个文件——`tcn.py`和`vmdpy.py`。分别来自于两个 pypi 包，`keras-tcn`和`vmdpy`。 由于它俩安装之后都是只有一个 py，所以直接拿出来了，防止以后找不到了。
      因此可以考虑不安装这两个 pypi 包，用时直接 import 这两个。用时要先把它两个从 reference 中移出来（因为 reference 设置为了`Excluded`）
      例如：移到`envirenment`中，`from environment.tcn import compiled_tcn`。
