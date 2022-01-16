import unittest
from base64 import b64encode
from unittest.mock import patch

import cv2
import numpy as np

from src.utils import prepare


class TestUtils(unittest.TestCase):

    @patch('src.utils.decode_image')
    def test_prepare(self, mock_settings):
        image = np.random.rand(256, 256, 3)
        mock_settings.return_value = image
        code = np.array(cv2.imencode('.png', image)[1]).tobytes()
        input_tensor = b64encode(code).decode('utf-8')
        result = prepare(input_tensor)
        self.assertEqual(result.shape, (256, 256, 3))
