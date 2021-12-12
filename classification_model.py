from tensorflow import keras
from tensorflow.keras import layers


class ClassificationModel(keras.Model):
    def __init__(
            self
    ):
        super(ClassificationModel, self).__init__()
        self.rescaling = layers.Rescaling(1. / 255)
        self.conv_1 = layers.Conv2D(32,
                                    kernel_size=(3, 3),
                                    activation="relu")
        self.conv_2 = layers.Conv2D(64,
                                    kernel_size=(3, 3),
                                    activation="relu")
        self.max_pooling = layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.5)
        self.dense = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.rescaling(inputs)
        x = self.conv_1(x)
        x = self.max_pooling(x)
        x = self.conv_2(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.dense(x)
