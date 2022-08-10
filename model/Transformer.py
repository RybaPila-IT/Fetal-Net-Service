import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from numpy import ndarray
from model.aliases import measurement_type, img_size


class ImageTransformer:
    """
    Class implementing utilities associated with image and tensor transformations.

    It is the utilit class for the FetalNet class. It implements several
    methods necessary for smooth image manipulation which is required
    during prediction generation pipeline.
    """
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    @staticmethod
    def image_to_tensor(image: ndarray) -> torch.tensor:
        """
        Transforms image to tensor which may be used by YNet network.
        """
        image_greyscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_pil = Image.fromarray(image_greyscale)
        image_tensor = ImageTransformer.transformations(image_pil).unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3, 4)
        return image_tensor

    @staticmethod
    def tensor_to_image(tensor: torch.tensor) -> ndarray:
        """
        Transforms tensor back to image representation
        """
        return tensor.squeeze(0).squeeze(0).detach().numpy()

    @staticmethod
    def resize_image(image: ndarray, size: img_size) -> ndarray:
        """
        Resizes the image to provided end size.
        """
        return cv2.resize(image, dsize=size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def reduce_noise_in_image(image: ndarray, body_part: str) -> ndarray:
        """
        Reduces the noise in image for better usage when applying the contour finding.

        Function will reduce the noise in image by applying the opening operator
        (erosion followed by dilation). At the end it will apply median blur
        operator.

        :param image: image which will be cleared.
        :param body_part: body part found in the image
        :return: image with reduced noise.
        """
        if body_part == 'femur':
            out_image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)[-1]
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            out_image = cv2.morphologyEx(out_image, cv2.MORPH_OPEN, kernel, iterations=2)
        elif body_part == 'head' or body_part == 'abdomen':
            out_image = cv2.threshold(image, 0.95, 1, cv2.THRESH_BINARY)[-1]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            out_image = cv2.morphologyEx(out_image, cv2.MORPH_OPEN, kernel, iterations=2)
        else:
            out_image = image
        out_image = out_image.astype(np.uint8)
        out_image = cv2.medianBlur(out_image, 13)
        return out_image

    @staticmethod
    def add_text_to_image(image: ndarray, body_part: str, prob: float, measurement: measurement_type) -> ndarray:
        """
        Adds text to the image for better readability.

        Text contains information about body part, its probability and its size.
        In the case of "head" class als the BPD information will be added.

        NOTE: Method changes the underlying image.

        :param image: image to which the text will be added
        :param body_part: body part found in the image.
        :param prob: probability of the correct classification.
        :param measurement: body part size measurement obtained from Measurer.
        :return: image with text added.
        """
        scale = 1e-3
        color_green = (0, 255, 0)
        img_width = image.shape[0]
        font_scale = img_width * scale
        bpd_spot = (round(img_width / 3), 35)
        body_part_spot = (round(img_width - img_width / 3), 35)
        measurement_spot = (round(img_width - img_width / 3), 75)

        body_part_size = measurement[0] if body_part == 'head' or body_part == 'abdomen' else measurement

        cv2.putText(
            image,
            text=f"{body_part} {prob * 100:.3f}%",
            org=body_part_spot,
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            lineType=cv2.LINE_AA,
            fontScale=font_scale,
            color=color_green
        )
        cv2.putText(
            image,
            text=f"{0.1 * body_part_size:.2f} cm",
            org=measurement_spot,
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            lineType=cv2.LINE_AA,
            fontScale=font_scale,
            color=color_green
        )

        if body_part == 'head':
            bpd = measurement[1]
            cv2.putText(
                image,
                text=f"BPD: {0.1 * bpd:.2f} cm",
                org=bpd_spot,
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                lineType=cv2.LINE_AA,
                fontScale=font_scale,
                color=color_green
            )

        return image
