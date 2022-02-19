from .. import networks
from .core import ClassificationModel
from typing import Any


class DkNet(ClassificationModel):

    def __init__(self, depth: int, **kwargs: Any) -> None:
        super(DkNet, self).__init__(net=networks.DkNet(depth=depth), depth=depth, **kwargs)
        self.save_hyperparameters()


class DkNetCL(ClassificationModel):

    def __init__(self, depth: int, **kwargs: Any) -> None:
        super(DkNetCL, self).__init__(net=networks.DkNetCL(depth=depth), depth=depth, **kwargs)
        self.save_hyperparameters()
