from .. import networks
from .core import ClassificationModel
from typing import Any


class D3ModNet(ClassificationModel):

    def __init__(self, **kwargs: Any) -> None:
        super(D3ModNet, self).__init__(net=networks.D3ModNet(), depth=3, **kwargs)
        self.save_hyperparameters()


class D3ModNetCL(ClassificationModel):

    def __init__(self, **kwargs: Any) -> None:
        super(D3ModNetCL, self).__init__(net=networks.D3ModNetCL(), depth=3, **kwargs)
        self.save_hyperparameters()
