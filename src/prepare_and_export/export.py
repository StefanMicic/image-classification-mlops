import tensorflow as tf

from utils import processing_pipeline


class ModelExporter:
    """
    A Class to prepare model for serving. Model will receive string as input, convert it to image
    and pass that image through the network.
    """

    def __init__(self, model_path: str, model_serving_path: str):
        """
        Creates instance of model exporter.

        Args:
            model_path : Path to model which will be converted to receive string.
            model_serving_path : Path where to export converted model.
        """
        self.model_path = model_path
        self.model_serving_path = model_serving_path

    def __call__(self, version: int = 1) -> None:
        """
        Adds string as input at the beginning of the model and export it for tensorflow serving.
        """
        model = tf.keras.models.load_model(self.model_path)

        model_input = tf.keras.Input(
            shape=[], batch_shape=None, dtype=tf.string, name="input_image")

        preprocess = tf.keras.layers.Lambda(
            processing_pipeline(),
            name="preprocess_image")(model_input)

        output = model(preprocess)

        model1 = tf.keras.Model(model_input, output)
        model1.save(f"{self.model_serving_path}/{version}")


def main():
    exporter = ModelExporter('my_model', 'serving')
    exporter()


if __name__ == "__main__":
    main()
