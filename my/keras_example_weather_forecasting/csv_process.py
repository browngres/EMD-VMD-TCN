import pandas as pd

from keras_example_weather_forecasting.data_info import titles, feature_keys, date_time_key

SAVE_PATH = '../../data/Jena Climate dataset/'
csv_name = "jena_climate_2009_2016.csv"

df = pd.read_csv(SAVE_PATH + csv_name)
# df.shape   # (420551,15)

# 降低训练难度，只要2016年的
#print(df.iloc[368291])  # 2016.01.01 00:00
df_2016 = df.iloc[368291:]
# print(df_2016.shape)  # (52260,15)
df_2016.to_csv(SAVE_PATH + 'jena_climate_2016.csv', index=False)

select_title = [0, 1, 5, 7, 8, 10, 11]
print("The selected parameters are:", ", ".join([titles[i] for i in select_title]), )
# The selected parameters are:
# Pressure, Temperature, Saturation vapor pressure, Vapor pressure deficit, Specific humidity, Airtight, Wind speed
selected_features = [feature_keys[i] for i in select_title]
features = df_2016[selected_features]
features.index = df_2016[date_time_key]
features.to_csv(SAVE_PATH + 'jena_climate_2016_selected.csv')