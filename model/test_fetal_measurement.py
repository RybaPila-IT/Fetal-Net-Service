import imagehash
import pytest
from PIL import Image

from model.fetal_measurement import FetalMeasurement


@pytest.mark.parametrize('image_path, prediction_path, body_part', [
    ('../data/samples/images/0427_abdomen_80.png', '../data/samples/predictions/0427_abdomen_80.png', 'abdomen'),
    ('../data/samples/images/0427_femur_43.png', '../data/samples/predictions/0427_femur_43.png', 'femur'),
    ('../data/samples/images/0427_head_203.png', '../data/samples/predictions/0427_head_203.png', 'head'),
    ('../data/samples/images/no_medical_image.png', '../data/samples/predictions/no_medical_image.png', 'unclassified'),
])
def test_predictions(image_path, prediction_path, body_part):
    measurer = FetalMeasurement(model_path='trained/weights.pt')
    # Maximum bits that could be different between the hashes.
    cutoff = 3
    # Start of the test.
    image = Image.open(image_path)
    predicted_body_part, result_image = measurer.get_prediction(image)
    prediction_img_hash = imagehash.average_hash(Image.open(prediction_path))
    result_img_hash = imagehash.average_hash(result_image)

    assert predicted_body_part == body_part
    assert abs(prediction_img_hash - result_img_hash) <= cutoff
