from ..typing import ImageBatch, Logits
import cl
import torch


class DkNet(torch.nn.Module):

    def __init__(self, depth: int) -> None:
        super(DkNet, self).__init__()
        layers = [
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        ]
        for _ in range(1, depth):
            layers.extend([
                torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            ])
        self.features = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(32768, 10)

    def forward(self, images: ImageBatch) -> Logits:
        x = self.features(images)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class DkNetCL(torch.nn.Module):

    def __init__(self, depth: int) -> None:
        super(DkNetCL, self).__init__()
        layers = [
            cl.Conv2d(3, 32, kernel_size=3, padding=1),
            cl.ReSPro(),
        ]
        for _ in range(1, depth):
            layers.extend([
                cl.Conv2d(32, 32, kernel_size=3, padding=1),
                cl.ReSPro(),
            ])
        self.features = cl.ConformalLayers(*layers)
        self.classifier = torch.nn.Linear(32768, 10)

    def forward(self, images: ImageBatch) -> Logits:
        x = self.features(images)
        x = torch.flatten(x, 1)
        return self.classifier(x)
