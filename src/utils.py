from typing import Any, Callable

import numpy as np
import tensorflow as tf
from tensorflow import Tensor


def decode_image(input_tensor: str) -> Tensor:
    """
    Decode base64 representation into image.

    Args:
        input_tensor: String representation of image.

    Returns:
        Tensor with image.
    """
    ret = tf.io.decode_png(input_tensor, channels=3)
    return tf.cast(ret, tf.float32)


def prepare(input_tensor: str) -> np.array:
    """
    Prepare input string for model by decoding, resizing and normalizing.

    Args:
        input_tensor: String representation of image.

    Returns:
        Prepared input for model.
    """
    ret = decode_image(input_tensor)
    ret = tf.image.resize(ret, (256, 256))
    ret = ret / 255
    return ret


def processing_pipeline() -> Callable[[str], Any]:
    def _process(input_tensor: str) -> Any:
        """
        Prepares every string representation of image by using prepare function.

        Args:
            input_tensor: String representation of image.
        """
        ret = tf.map_fn(
            prepare,
            input_tensor,
            dtype=tf.float32)
        ret.set_shape([None, None, None, 3])
        return ret

    return _process
