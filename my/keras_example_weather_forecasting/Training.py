from matplotlib import pyplot as plt
from tensorflow import keras

from keras_example_weather_forecasting.data_preprocessing import dataset_train, dataset_val, inputs
from keras_example_weather_forecasting.hyperparamter import learning_rate, epochs

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
# 描述模型
model.summary()

# 存储模型并早停
path_checkpoint = "weather_forecasting_model_checkpoint.h5"
callbacks = [
    keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        # save_weights_only=True,
        save_best_only=True,
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5)
]

# fit
history = model.fit(dataset_train, epochs=epochs, validation_data=dataset_val, callbacks=callbacks)


def visualize_loss(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history)
