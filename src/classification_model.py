import tensorflow as tf
from tensorflow import keras


class ClassificationModel(keras.Model):
    """A model for test classification. It should tell whether the test is positive or not."""

    def __init__(self):
        """Creates instance of test classification model."""
        super(ClassificationModel, self).__init__()

        size = 256
        base_model = keras.applications.MobileNetV2(
            weights="imagenet",
            input_shape=(size, size, 3),
            include_top=False,
        )

        base_model.trainable = False

        self.base_model = base_model
        self.avgpool_1 = keras.layers.GlobalAveragePooling2D()
        self.dense_1 = keras.layers.Dense(256, activation='relu')
        self.dropout_1 = keras.layers.Dropout(0.2)
        self.dense_2 = keras.layers.Dense(128, activation='relu')
        self.dropout_2 = keras.layers.Dropout(0.2)
        self.dense_3 = keras.layers.Dense(64, activation='relu')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass in neural network.

        Args:
            inputs: Input tensor.

        Returns:
            Prediction given by model.
        """
        x = self.base_model(inputs)
        x = self.avgpool_1(x)
        x = self.dense_1(x)

        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        x = self.dense_3(x)

        return x
