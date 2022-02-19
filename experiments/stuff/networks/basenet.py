from ..typing import ImageBatch, Logits
import cl
import torch


class BaseLinearNet(torch.nn.Module):

    def __init__(self) -> None:
        super(BaseLinearNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = torch.nn.Linear(7200, 10)

    def forward(self, images: ImageBatch) -> Logits:
        x = self.features(images)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class BaseReLUNet(torch.nn.Module):
    
    def __init__(self) -> None:
        super(BaseReLUNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = torch.nn.Linear(7200, 10)

    def forward(self, images: ImageBatch) -> Logits:
        x = self.features(images)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class BaseReSProNet(torch.nn.Module):

    def __init__(self) -> None:
        super(BaseReSProNet, self).__init__()
        self.features = cl.ConformalLayers(
            cl.Conv2d(3, 32, kernel_size=3),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = torch.nn.Linear(7200, 10)

    def forward(self, images: ImageBatch) -> Logits:
        x = self.features(images)
        x = torch.flatten(x, 1)
        return self.classifier(x)
