import torch
import torchvision.transforms as transforms
from PIL import Image

from model.fetal_net import YNet

# Type representing the prediction obtained from the measurer.
Prediction = tuple[str, Image]


class FetalMeasurement:
    def __init__(self, model_path: str,
                 end_img_size: tuple[int, int] = (512, 512),
                 mask_blend_strength: float = 0.24):
        self.model_path = model_path
        self.model = YNet(
            input_channels=1,
            output_channels=64,
            n_class=1
        )
        self.model.load_state_dict(
            torch.load(self.model_path, map_location="cpu")
        )
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.end_img_size = end_img_size
        self.mask_blend_strength = mask_blend_strength

    def get_prediction(self, input_img: Image) -> Prediction:
        """
        Obtain a prediction result.

        Function performs the prediction on provided image by using the Fetal-Net
        neural network.

        :param: input_img: PIL image on which prediction will be made
        :return: classified body part, result image in PIL format
        """
        cls, mask = self.__make_prediction(input_img)
        body_part = self.__classify_body_part(cls)
        mask_img = self.__prepare_mask(mask)
        result_img = self.__obtain_final_image(input_img, mask_img)
        return body_part, result_img

    def __make_prediction(self, image: Image) -> tuple[torch.tensor, torch.tensor]:
        # Convert image into greyscale and then reshape it into tensor meaning
        # batch size, number of frames, channels, frame size.
        # Eventually feed the FetalNet with that tensor.
        return self.model.forward(
            self.transforms(image.convert('L')).reshape([1, 1, 1, 224, 224])
        )

    @staticmethod
    def __classify_body_part(cls: torch.tensor) -> str:
        cls = torch.softmax(cls, dim=1)
        cls_max = torch.argmax(cls, dim=1)
        if cls_max == torch.tensor([0]):
            return 'head'
        if cls_max == torch.tensor([1]):
            return 'abdomen'
        if cls_max == torch.tensor([2]):
            return 'femur'
        return 'unclassified'

    @staticmethod
    def __prepare_mask(mask: torch.tensor) -> Image:
        data = mask.round().squeeze(0).cpu().data
        mask_img = transforms.ToPILImage()(data).convert('RGB')
        r, g, b = mask_img.split()
        # Changing channels in order to make mask color stand out.
        # Now it will be green.
        r = r.point(lambda i: i * 0)
        g = g.point(lambda i: i * 5)
        b = b.point(lambda i: i * 0)

        return Image.merge('RGB', (r, g, b))

    def __obtain_final_image(self, input_image: Image, mask_image: Image) -> Image:
        return Image.blend(
            input_image.convert('RGB').resize(self.end_img_size),
            mask_image.convert('RGB').resize(self.end_img_size),
            self.mask_blend_strength
        )
