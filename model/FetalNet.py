import torch
from numpy import ndarray

from model.Transformer import ImageTransformer
from model.network.YNet import YNet
from model.Classifier import BodyPartClassifier
from model.Measurer import BodyPartMeasurer
from model.aliases import prediction


class FetalNet:
    """
    Main class implementing whole prediction pipeline.
    """

    def __init__(self, model_path: str):
        self.model = YNet()
        self.transformer = ImageTransformer()
        self.classifier = BodyPartClassifier()
        self.measurer = BodyPartMeasurer()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def predict(self, image: ndarray, attributes: dict) -> prediction:
        """
        Obtain a prediction result.

        Function performs the prediction on provided image by using the Fetal-Net
        neural network.

        :param: input_img: PIL image on which prediction will be made
        :return: classified body part, body part size, result image in PIL format
        """
        pixel_spacing, image_size = attributes['pixel_spacing'], attributes['image_size']
        img_resized = self.transformer.resize_image(image, image_size)
        tensor_img = self.transformer.image_to_tensor(image)
        cls, mask = self.model.forward(tensor_img)
        body_part, prob = self.classifier.classify_body_part(cls)
        if body_part == 'unclassified':
            return body_part, '0 cm', image
        mask = self.transformer.tensor_to_image(mask)
        mask = self.transformer.resize_image(mask, image_size)
        mask = self.transformer.reduce_noise_in_image(mask, body_part)
        measurement, combined_img = self.measurer.take_measurement(img_resized, mask, body_part, pixel_spacing)
        result_img = self.transformer.add_text_to_image(combined_img, body_part, prob, measurement)
        measurement = self.measurer.measurement_to_text(measurement)
        return body_part, measurement, result_img
