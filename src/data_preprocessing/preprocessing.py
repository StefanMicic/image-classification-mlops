import os

import cv2
import tensorflow as tf

from src.data_preprocessing.lakefs import LakeFSManipulator


class DataPipeline:
    """A class for preprocessing data and for data versioning using lakeFS."""

    def __init__(self, dataset_directory: str, input_shape: int = 256):
        """
        Creates an instance with desired image shapes and path to the dictionary containing dataset.

        Args:
            dataset_directory: Path to the dataset
            input_shape: Desired image shape
        """
        self.dataset_directory = dataset_directory
        self.classes = os.listdir(dataset_directory)
        self.input_size = (input_shape, input_shape)

    def prepare_dataset(self) -> None:
        """Resizes every image in every class inside dataset."""
        for class_name in self.classes:
            folder_path = f"{self.dataset_directory}/{class_name}"
            for image_name in os.listdir(folder_path):
                image_path = f"{folder_path}/{image_name}"
                image = cv2.imread(image_path)
                resized = cv2.resize(image, self.input_size)
                cv2.imwrite(image_path, resized)

    def get_dataset(self) -> tf.data.Dataset:
        """Generates dataset for training from directory.

        Returns:
                Data generator for training.
        """
        return tf.keras.utils.image_dataset_from_directory(self.dataset_directory, batch_size=2)

    def commit_data(self) -> None:
        """
        Git-like behaviour for data versioning. Creates branch if not exists, add new and modified
        files, commits them and merge with desired branch.
        """
        lakefs_manipulator = LakeFSManipulator()
        lakefs_manipulator.create_branch()
        lakefs_manipulator.add_files_from_dir(self.dataset_directory, self.classes)
        lakefs_manipulator.commit_changes()
        lakefs_manipulator.merge()
