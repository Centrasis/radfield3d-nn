from torch import nn, Tensor


class EncodingBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__encoded_dims = None

    def __get_encoded_dims(self) -> int:
        if self.__encoded_dims is None:
            self.__encoded_dims = self.calc_encoded_dim()
        return self.__encoded_dims

    encoded_dims = property(__get_encoded_dims)

    def calc_encoded_dim(self) -> int:
        """
        Calculate the number of output dimensions the encoding.
        """
        ...

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the encoding.
        """
        ...
