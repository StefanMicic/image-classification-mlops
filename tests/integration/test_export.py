import os.path
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from src.classification_model import ClassificationModel
from src.prepare_and_export.export import ModelExporter


class TestExport(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.test_dir, 'cat'))
        os.mkdir(os.path.join(self.test_dir, 'dog'))
        image = np.random.rand(64, 64, 3)
        cv2.imwrite(os.path.join(self.test_dir, 'cat/test.jpg'), image)
        cv2.imwrite(os.path.join(self.test_dir, 'dog/test.jpg'), image)
        self.create_model()

    def create_model(self):
        model = ClassificationModel()
        model.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        model.compute_output_shape(input_shape=(None, 256, 256, 3))
        model.save(os.path.join(self.test_dir, 'model'))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_training(self):
        exporter = ModelExporter(os.path.join(self.test_dir, 'model'), os.path.join(self.test_dir, 'serving'))
        exporter()
        self.assertEqual(os.path.exists(os.path.join(self.test_dir, os.path.join(self.test_dir, 'serving'))), True)
