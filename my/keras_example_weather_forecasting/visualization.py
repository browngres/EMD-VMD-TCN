import pandas as pd
from matplotlib import pyplot as plt

from keras_example_weather_forecasting.data_info import date_time_key, feature_keys, titles, colors

SAVE_PATH = '../../data/Jena Climate dataset/'
csv_name = "jena_climate_2016.csv"
df = pd.read_csv(SAVE_PATH+csv_name)


def only_date(x):
    return x[:5]


def show_raw_visualization(data):
    # 坐标轴只要日期，不要年份
    time_data = data[date_time_key].apply(only_date)
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k")
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        t_data = data[key]
        t_data.index = time_data
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=colors[i % (len(colors))],
            lw=0.5,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        #ax.legend([titles[i]])
    plt.tight_layout()
    plt.show()


show_raw_visualization(df)


"""
This heat map shows the correlation between different features.
"""


def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=8, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=8)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()


show_heatmap(df)
