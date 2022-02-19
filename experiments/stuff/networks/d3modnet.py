from ..typing import ImageBatch, Logits
import cl
import torch


class D3ModNet(torch.nn.Module):
    
    def __init__(self) -> None:
        super(D3ModNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = torch.nn.Linear(128, 10)

    def forward(self, images: ImageBatch) -> Logits:
        x = self.features(images)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class D3ModNetCL(torch.nn.Module):

    def __init__(self) -> None:
        super(D3ModNetCL, self).__init__()
        self.features = cl.ConformalLayers(
            cl.Conv2d(3, 32, kernel_size=3),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
            cl.Conv2d(32, 32, kernel_size=3),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
            cl.Conv2d(32, 32, kernel_size=3),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = torch.nn.Linear(128, 10)

    def forward(self, images: ImageBatch) -> Logits:
        x = self.features(images)
        x = torch.flatten(x, 1)
        return self.classifier(x)
