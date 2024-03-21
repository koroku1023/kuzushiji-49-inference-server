from torch import nn
from torchvision.models import densenet121
from torchvision.models.densenet import DenseNet121_Weights


class DenseNet_121(nn.Module):

    def __init__(self, num_classes):

        super(DenseNet_121, self).__init__()
        self.densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.densenet(x)
