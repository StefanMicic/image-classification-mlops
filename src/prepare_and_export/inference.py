from base64 import b64encode
from typing import Any, Dict

import cv2
import numpy as np
import requests


def encode_image(img_path: str) -> str:
    """ Encode image to base64 representation.

    Args:
        img_path: Path to the image.

    Returns:
        Base64 representation of image.
    """
    image = cv2.imread(img_path)
    code = np.array(cv2.imencode('.jpg', image)[1]).tobytes()
    return b64encode(code).decode('utf-8')


def request_data(net_input: str) -> Dict[str, Any]:
    """ Setups data dict for post request.

    Args:
        net_input: Base64 representation of image.

    Returns:
        Dictionary for request.
    """
    ret = {'signature_name': "serving_default", 'instances': list()}
    ret['instances'].append({
        "input_image": {"b64": net_input}
    })
    return ret


def main():
    input_data = encode_image("dataset_cat_vs_dog/dog/images.jpeg")

    predict_url = f'http://localhost:8501'
    url = (
        f'{predict_url}/v1/models/dog_detector:predict'
    )
    req = requests.post(
        url,
        json=request_data(input_data),
        timeout=500
    )
    result = req.json()
    predictions = result['predictions']
    print(predictions)


if __name__ == "__main__":
    main()
