import torch
from model.aliases import classification


class BodyPartClassifier:
    """
    Class enabling the classification process of the body part presented at the provided image.
    """

    @staticmethod
    def classify_body_part(cls: torch.tensor) -> classification:
        """
        Enables to classify the body part into 1 of 4 groups: "femur", "abdomen", "head" and "unclassified".

        Method classifies the output of YNet network by obtaining the most probable
        group from tensor of classification. It also calculates the probability of
        correct classification (network belief).

        :param cls: tensor representing the network classification output.
        :return: tuple containing the classified body part and probability.
        """
        cls = torch.softmax(cls, dim=1)
        cls_max = torch.argmax(cls, dim=1)
        probability = torch.max(cls).item()
        if probability < 0.25:
            return 'unclassified', probability
        if cls_max == torch.tensor([0]):
            return 'head', probability
        if cls_max == torch.tensor([1]):
            return 'abdomen', probability
        if cls_max == torch.tensor([2]):
            return 'femur', probability
        return 'unclassified', probability
