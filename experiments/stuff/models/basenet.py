from .. import networks
from .core import ClassificationModel
from typing import Any


class BaseLinearNet(ClassificationModel):

    def __init__(self, **kwargs: Any) -> None:
        super(BaseLinearNet, self).__init__(net=networks.BaseLinearNet(), depth=1, **kwargs)
        self.save_hyperparameters()


class BaseReLUNet(ClassificationModel):

    def __init__(self, **kwargs: Any) -> None:
        super(BaseReLUNet, self).__init__(net=networks.BaseReLUNet(), depth=1, **kwargs)
        self.save_hyperparameters()


class BaseReSProNet(ClassificationModel):

    def __init__(self, **kwargs: Any) -> None:
        super(BaseReSProNet, self).__init__(net=networks.BaseReSProNet(), depth=1, **kwargs)
        self.save_hyperparameters()
