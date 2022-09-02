from .. import networks
from .core import ClassificationModel
from typing import Any


class LeNet(ClassificationModel):

    def __init__(self, **kwargs: Any) -> None:
        super(LeNet, self).__init__(net=networks.LeNet(), depth=2, **kwargs)
        self.save_hyperparameters()


class LeNetCL(ClassificationModel):

    def __init__(self, **kwargs: Any) -> None:
        super(LeNetCL, self).__init__(net=networks.LeNetCL(), depth=2, **kwargs)
        self.save_hyperparameters()
