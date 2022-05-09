import imagehash
from PIL import Image

from model.fetal_measurement import FetalMeasurement


class TestFetalMeasurement:
    def test_predictions(self):
        measurer = FetalMeasurement(model_path='../model/trained/weights.pt')
        image_paths = [
            '../data/samples/images/0427_abdomen_80.png',
            '../data/samples/images/0427_femur_43.png',
            '../data/samples/images/0427_head_203.png',
            '../data/samples/images/no_medical_image.png'
        ]
        predictions_paths = [
            '../data/samples/predictions/0427_abdomen_80.png',
            '../data/samples/predictions/0427_femur_43.png',
            '../data/samples/predictions/0427_head_203.png',
            '../data/samples/predictions/no_medical_image.png'
        ]
        body_parts = [
            'abdomen',
            'femur',
            'head',
            'unclassified'
        ]
        # Maximum bits that could be different between the hashes.
        cutoff = 3

        for image_path, body_part, prediction in zip(image_paths, body_parts, predictions_paths):
            image = Image.open(image_path)
            predicted_body_part, result_image = measurer.get_prediction(image)

            prediction_img_hash = imagehash.average_hash(Image.open(prediction))
            result_img_hash = imagehash.average_hash(result_image)

            assert predicted_body_part == body_part
            assert abs(prediction_img_hash - result_img_hash) <= cutoff
