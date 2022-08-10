import base64
import io
import os
import cv2
import numpy as np
import torch

from numpy import ndarray
from dotenv import load_dotenv
from PIL import Image
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from binascii import Error

from model.FetalNet import FetalNet
from model.aliases import prediction

torch.set_grad_enabled(False)
torch.set_flush_denormal(True)


class Request(BaseModel):
    photo: str
    attributes: dict


class Response(BaseModel):
    photo: str
    prediction: dict


MODEL_PATH = 'model/trained/weights.pt'
ACCESS_TOKEN_ENV_KEY = 'ACCESS_TOKEN'

# Preparing the environment of the service.
load_dotenv()

security = HTTPBearer()
app = FastAPI()
fetal_net = FetalNet(MODEL_PATH)


@app.get('/')
def main() -> dict:
    return {
        'message': 'Welcome to Fetal-Net Service'
    }


@app.post('/predict')
def predict(req: Request, credentials: HTTPAuthorizationCredentials = Security(security)):
    if not __valid_credentials(credentials.credentials):
        raise HTTPException(status.HTTP_403_FORBIDDEN, 'Invalid access token')
    image = __decode_image(req)
    attributes = __obtain_attributes(req)
    body_part, body_part_size, image = __predict(image, attributes)
    result_img_bytes = __encode_image(image)
    return Response(
        photo=result_img_bytes,
        prediction={
            'body_part': body_part,
            'size': body_part_size
        }
    )


def __valid_credentials(credentials: str) -> bool:
    return credentials == os.getenv(ACCESS_TOKEN_ENV_KEY)


def __obtain_attributes(req: Request) -> dict:
    if 'pixel_spacing' not in req.attributes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'Missing pixel spacing attribute for making prediction')
    if 'image_size' not in req.attributes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'Missing image size attribute for making prediction')
    return req.attributes


def __decode_image(req: Request) -> ndarray:
    try:
        io_buffer = io.BytesIO(base64.b64decode(req.photo))
        file_bytes = np.asarray(bytearray(io_buffer.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'Submitted file is corrupted')


def __predict(image: Image, attributes: dict) -> prediction:
    try:
        return fetal_net.predict(image, attributes)
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f'Error while performing the prediction: {e}')


def __encode_image(image: ndarray) -> bytes:
    try:
        is_success, buffer = cv2.imencode(".png", image)
        if not is_success:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, 'Saving file into buffer was not successful')
        io_buffer = io.BytesIO(buffer)
        return base64.b64encode(io_buffer.getvalue())
    except OSError as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f'Error while saving file into BytesIO: {e}')
