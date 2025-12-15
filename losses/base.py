from torch import Tensor, nn
from rftypes import TrainingInputData


class Loss(nn.Module):
    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        raise NotImplementedError("This method must be implemented in a subclass.")
