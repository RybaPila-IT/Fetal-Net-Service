import base64
import io

from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

from model.FetalMeasurement import FetalMeasurement


class ImageData(BaseModel):
    pixels: str


fetal_measurement = FetalMeasurement(
    model_path='model/trained/weights.pt'
)

# TODO (radek.r) Add authorization
# TODO (radek.r) Add image receiving
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
    img_pixels = base64.b64decode(image_data.pixels)
    buffer = io.BytesIO(img_pixels)
    return Image.open(buffer)


def __predict(image: Image) -> tuple[str, Image]:
    return fetal_measurement.get_prediction(image)


def __encode_image(image: Image) -> bytes:
    buffered = io.BytesIO()
    image.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue())
