import os.path
import shutil
import tempfile
import unittest
from base64 import b64encode

import cv2
import numpy as np

from prepare_and_export.inference import encode_image, request_data


class TestInference(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_encode_image(self):
        image = np.random.rand(64, 64, 3)
        cv2.imwrite(os.path.join(self.test_dir, 'test.jpg'), image)
        encoded = encode_image(os.path.join(self.test_dir, 'test.jpg'))

        code = np.array(cv2.imencode('.jpg', cv2.imread(os.path.join(self.test_dir, 'test.jpg')))[1]).tobytes()
        self.assertEqual(b64encode(code).decode('utf-8'), encoded)

    def test_request_data(self):
        result = request_data("test")
        self.assertEqual(result['instances'][0]['input_image']['b64'], 'test')
