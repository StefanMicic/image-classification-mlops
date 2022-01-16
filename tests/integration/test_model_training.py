import os.path
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from src.classification_model import ClassificationModel
from src.data_preprocessing.preprocessing import DataPipeline


class TestTraining(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.test_dir, 'cat'))
        os.mkdir(os.path.join(self.test_dir, 'dog'))
        image = np.random.rand(64, 64, 3)
        cv2.imwrite(os.path.join(self.test_dir, 'cat/test.jpg'), image)
        cv2.imwrite(os.path.join(self.test_dir, 'dog/test.jpg'), image)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_training(self):
        data_pipeline = DataPipeline(self.test_dir, 128)
        data_pipeline.prepare_dataset()
        train_dataset = data_pipeline.get_dataset()

        model = ClassificationModel()
        model.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        model.fit(train_dataset,
                  batch_size=1,
                  epochs=1)
        model.save(os.path.join(self.test_dir, 'model'))
        self.assertEqual(os.path.exists(os.path.join(self.test_dir, 'model')), True)
