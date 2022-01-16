import argparse

import mlflow

from classification_model import ClassificationModel
from data_preprocessing.preprocessing import DataPipeline
from prepare_and_export.export import ModelExporter


def log_model(dt_params, metrics, tf_model) -> None:
    """
    Logs models metrics and parameters to MLFlow.

    Args:
        dt_params: Parameters for logging.
        metrics: Metrics for logging.
        tf_model: Model to be saved before converting for tensorflow serving.
    """
    with mlflow.start_run(run_name='model subclassing'):
        mlflow.log_params(dt_params)
        mlflow.log_metric('Accuracy', metrics['accuracy'])
        mlflow.log_metric('Loss', metrics['loss'])

        mlflow.keras.log_model(tf_model, "models")


def pipeline():
    parser = argparse.ArgumentParser(description="Image extraction")
    parser.add_argument("--input_shape", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="my_model")
    args = parser.parse_args()

    mlflow.set_experiment('Cat_vs_Dog')

    data_pipeline = DataPipeline("dataset_cat_vs_dog")
    data_pipeline.prepare_dataset()
    train_dataset = data_pipeline.get_dataset()
    data_pipeline.commit_data()

    model = ClassificationModel()
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    hist = model.fit(train_dataset,
                     batch_size=args.batch_size,
                     epochs=args.epochs)
    log_model({}, {'accuracy': hist.history['accuracy'][-1], 'loss': hist.history['loss'][-1]}, model)
    model.save(args.model_path)
    exporter = ModelExporter(args.model_path, 'serving')
    exporter()


if __name__ == "__main__":
    pipeline()
