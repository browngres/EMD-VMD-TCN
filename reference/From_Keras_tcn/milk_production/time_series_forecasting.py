# monthly-milk-production-pounds-per-cow-jan-62-dec-75
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tcn import TCN
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

##
# It's a very naive (toy) example to show how to do time series forecasting.
# - There are no training-testing sets here. Everything is training set for simplicity.
# - There is no input/output normalization.
# - The model is simple.
##

milk = pd.read_csv('monthly-milk-production.csv', index_col=0, parse_dates=True)
timestep = 12  # months.
milk = milk.values  # (168,1)

'''
for i in range(timestep, train_end):
    train_x.append(milk[i - timestep:i])
    train_y.append(milk[i])

for i in range(train_end - timestep):
    train_x.append(milk[i:i + timestep])
    train_y.append(milk[i + timestep])
'''
train_end = int(0.91 * len(milk))  # 最后剩13个用于测试   train_end: 152
train_x, train_y = [], []
for i in range(train_end - timestep):
    train_x.append(milk[i:i + timestep])
    train_y.append(milk[i + timestep])
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = milk[train_end:train_end + timestep]
test_y = milk[train_end + timestep]

print(train_x.shape)  # (140, 12, 1)
print(train_y.shape)  # (140, 1)
print(test_x.shape)  # (12, 1)
print(test_y.shape)  # (1)

# noinspection PyArgumentEqualDefault
model = Sequential([
    TCN(input_shape=(timestep, 1),
        #nb_filters=24,
        kernel_size=2,
        # dilations=[2 ** i for i in range(4)],
        use_skip_connections=False,
        use_batch_norm=False,
        use_weight_norm=False,
        use_layer_norm=False
        ),
    Dense(1, activation='linear')
])
model.compile('adam', 'mae')
# input_shape: (None, 12, 1)  output_shape: (None, 1)

model.fit(train_x, train_y, epochs=100, verbose=2)
index = range(0, len(milk))
plt.title('Monthly Milk Production (in pounds)')

plt.plot(index[:train_end], milk[:train_end], ".-", label="History")
plt.plot(index[train_end:train_end + timestep], test_x, ".:c", label="Train_X")
plt.plot(index[train_end + timestep], test_y, "gx", label="True Future")
plt.plot(index[train_end + timestep], model.predict(np.array([test_x])), "ro", label="Model Prediction", alpha=0.5)
plt.grid(axis='x')
plt.xlim(0, 180)
plt.legend()
plt.show()
