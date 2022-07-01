import base64
import io
import os

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from binascii import Error

from model.fetal_measurement import FetalMeasurement, Prediction


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
measurer = FetalMeasurement(MODEL_PATH)


@app.get('/')
def main() -> dict:
    return {
        'message': 'Welcome to Fetal-Net Service'
    }


@app.post('/predict')
def predict(req: Request, credentials: HTTPAuthorizationCredentials = Security(security)):
    if not __valid_credentials(credentials.credentials):
        raise HTTPException(status.HTTP_403_FORBIDDEN, 'Invalid access token')
    # Start of the prediction pipeline.
    image = __decode_image(req)
    body_part, result_img = __predict(image)
    result_img_bytes = __encode_image(result_img)
    # TODO (radek.r) Add also body part size prediction to response.
    return Response(
        photo=result_img_bytes,
        prediction={
            'body_part': body_part
        }
    )


def __valid_credentials(credentials: str) -> bool:
    return credentials == os.getenv(ACCESS_TOKEN_ENV_KEY)


def __decode_image(req: Request) -> Image:
    try:
        return Image.open(
            io.BytesIO(
                base64.b64decode(req.photo)
            )
        )
    except Error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'Submitted file is corrupted')
    except UnidentifiedImageError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'File is not a valid image')


def __predict(image: Image) -> Prediction:
    try:
        return measurer.get_prediction(image)
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f'Error while performing the prediction: {e}')


def __encode_image(image: Image) -> bytes:
    try:
        buffered = io.BytesIO()
        image.save(buffered, 'PNG')
        return base64.b64encode(buffered.getvalue())
    except OSError as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f'Error while saving file into BytesIO: {e}')
