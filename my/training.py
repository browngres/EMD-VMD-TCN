from matplotlib import pyplot as plt
from tcn import compiled_tcn
from tensorflow import keras

from data_set import Dataset
from hyperparameter import EPOCHS, TIMESTEP, SAVED_MODEL_PATH, COL

if __name__ == '__main__':
    # 准备数据集
    d = Dataset()
    train_data = d.get_train_x_y(col=COL)
    val_data = d.get_val_x_y(col=COL)

    # 模型
    model = compiled_tcn(num_feat=1,
                         nb_filters=48,
                         num_classes=0,
                         kernel_size=3,
                         dilations=[2 ** i for i in range(6)],
                         nb_stacks=1,
                         max_len=TIMESTEP,
                         output_len=1,
                         use_skip_connections=True,
                         return_sequences=False,
                         regression=True,
                         dropout_rate=0,
                         activation='relu',
                         opt='adam',
                         name='TEST_TCN',
                         lr=0.002,
                         use_batch_norm=False,
                         use_layer_norm=False,
                         use_weight_norm=True,
                         )
    # print(model.get_config())
    # model.summary()
    print('receptive_field: ', model.layers[1].receptive_field)

    checkpoint = SAVED_MODEL_PATH + "feature_{}_col_{}_best_model_{}.h5".format(d.feature_num, COL, d.tpb)
    callbacks = [
        # 存储最优模型
        keras.callbacks.ModelCheckpoint(checkpoint, save_best_only=True, monitor="val_loss", verbose=1),
        # 稳定时减小学习率
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=0.0001),
        # EarlyStop
        keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=25, verbose=1),
    ]

    history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data, callbacks=callbacks)


    def visualize_loss(history):
        """
        绘制训练过程的loss
        :param history: 训练的过程。( history = model.fit(..) )
        :return:
        """
        loss = history.history["loss"][1:]  # 不要第一个。第一个loss太大了，后面的就成了直线。
        val_loss = history.history["val_loss"][1:]
        plt.figure()
        plt.plot(range(len(loss)), loss, "b", label="Training loss")
        plt.plot(range(len(loss)), val_loss, "r:", label="Validation loss")
        plt.ylim((0, 0.2))
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    visualize_loss(history)
