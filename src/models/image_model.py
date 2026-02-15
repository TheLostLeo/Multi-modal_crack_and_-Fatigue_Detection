import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def get_model(name, num_classes, pretrained=True):

    if name == "resnet18":
        if pretrained:
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18(weights=None)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("Unsupported model")

    return model

