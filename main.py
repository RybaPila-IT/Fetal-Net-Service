import base64
import io
import os

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from binascii import Error

from model.fetal_measurement import FetalMeasurement


class ImageData(BaseModel):
    pixels: str


load_dotenv()

fetal_measurement = FetalMeasurement(
    model_path='model/trained/weights.pt'
)

# TODO (rade.r) Add tests


security = HTTPBearer()
app = FastAPI()


@app.post('/predict')
async def predict(image_data: ImageData,
                  credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    __validate_token(credentials.credentials)
    image = __decode_image(image_data)
    body_part, result_img = __predict(image)
    result_img_bytes = __encode_image(result_img)
    return {
        'body_part': body_part,
        'image_bytes': result_img_bytes
    }


def __validate_token(token):
    if token != os.getenv('ACCESS_TOKEN'):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail='Invalid access token')


def __decode_image(image_data: ImageData) -> Image:
    try:
        return Image.open(
            io.BytesIO(
                base64.b64decode(image_data.pixels)
            )
        )
    except Error:
        raise HTTPException(status_code=400, detail='Submitted file is corrupted')
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail='File is not a valid image')


def __predict(image: Image) -> tuple[str, Image]:
    try:
        return fetal_measurement.get_prediction(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error while performing the prediction: {e}')


def __encode_image(image: Image) -> bytes:
    try:
        buffered = io.BytesIO()
        image.save(buffered, format='PNG')
        return base64.b64encode(buffered.getvalue())
    except OSError:
        raise HTTPException(status_code=500, detail='Error while saving file into BytesIO')
