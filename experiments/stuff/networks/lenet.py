from ..typing import ImageBatch, Logits
import cl
import torch


class LeNet(torch.nn.Module):

    def __init__(self) -> None:
        super(LeNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(400, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10),
        )


    def forward(self, images: ImageBatch) -> Logits:
        x = self.features(images)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class LeNetCL(torch.nn.Module):

    def __init__(self) -> None:
        super(LeNetCL, self).__init__()
        self.features = cl.ConformalLayers(
            cl.Conv2d(3, 6, kernel_size=5),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
            cl.Conv2d(6, 16, kernel_size=5),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(400, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10),
        )

    def forward(self, images: ImageBatch) -> Logits:
        x = self.features(images)
        x = torch.flatten(x, 1)
        return self.classifier(x)
