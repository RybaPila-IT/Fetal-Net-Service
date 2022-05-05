"""Prediction and automatic measurement of fetal body parts"""
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model.FetalNet import YNet


class FetalMeasurement:
    """FetalMeasurement class."""

    def __init__(self, model_path):
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
        self.end_size = (512, 512)

    def get_prediction(self, image_name):
        image = Image.open(image_name).convert("L")
        x = self.transforms(image)
        x = x.reshape([1, 1, 1, 224, 224])
        cls, mask = self.model.forward(x)
        body_part = FetalMeasurement.__classify(cls)
        print(body_part)

        data = mask.round().squeeze(0).cpu().data
        mask = transforms.ToPILImage()(data).resize(self.end_size)
        mask.show()

    @staticmethod
    def __classify(cls):
        cls = torch.softmax(cls, dim=1)
        cls_max = torch.argmax(cls, dim=1)
        if cls_max == torch.tensor([0]):
            return 'head'
        if cls_max == torch.tensor([1]):
            return 'abdomen'
        return 'femur'


if __name__ == "__main__":
    fetal_measurement = FetalMeasurement(model_path='trained/weights.pt')
    fetal_measurement.get_prediction(image_name='../0427_femur_43.png')
