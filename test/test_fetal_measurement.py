import imagehash
import pytest
from PIL import Image

from model.fetal_measurement import FetalMeasurement


@pytest.mark.parametrize('image_path, prediction_path, body_part', [
    ('test/data/samples/images/0427_abdomen_80.png', 'test/data/samples/predictions/0427_abdomen_80.png', 'abdomen'),
    ('test/data/samples/images/0427_femur_43.png', 'test/data/samples/predictions/0427_femur_43.png', 'femur'),
    ('test/data/samples/images/0427_head_203.png', 'test/data/samples/predictions/0427_head_203.png', 'head'),
    ('test/data/samples/images/no_medical_image.png', 'test/data/samples/predictions/no_medical_image.png', 'unclassified'),
])
def test_predictions(image_path, prediction_path, body_part):
    measurer = FetalMeasurement(model_path='model/trained/weights.pt')
    # Maximum bits that could be different between the hashes.
    cutoff = 3
    # Start of the test.
    image = Image.open(image_path)
    predicted_body_part, result_image = measurer.get_prediction(image)
    prediction_img_hash = imagehash.average_hash(Image.open(prediction_path))
    result_img_hash = imagehash.average_hash(result_image)

    assert predicted_body_part == body_part
    assert abs(prediction_img_hash - result_img_hash) <= cutoff
