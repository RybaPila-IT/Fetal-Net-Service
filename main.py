import base64
import io

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from binascii import Error

from model.FetalMeasurement import FetalMeasurement


class ImageData(BaseModel):
    pixels: str


fetal_measurement = FetalMeasurement(
    model_path='model/trained/weights.pt'
)

# TODO (radek.r) Add authorization
# TODO (radek.r) Add exceptions handling


app = FastAPI()


@app.post('/predict')
async def predict(image_data: ImageData) -> dict:
    image = __decode_image(image_data)
    body_part, result_img = __predict(image)
    result_img_bytes = __encode_image(result_img)
    return {
        'body_part': body_part,
        'image_bytes': result_img_bytes
    }


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
        print(e)
        raise HTTPException(status_code=500, detail='Error while performing the prediction')


def __encode_image(image: Image) -> bytes:
    try:
        buffered = io.BytesIO()
        image.save(buffered, format='PNG')
        return base64.b64encode(buffered.getvalue())
    except OSError:
        raise HTTPException(status_code=500, detail='Error while saving file into BytesIO')
