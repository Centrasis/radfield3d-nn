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

    @property
    def region_state_dims(self) -> int:
        """Number of floats this encoder needs to be configured for a query region (0 = the
        encoder is region-agnostic and ignores the region entirely)."""
        return 0

    def compute_region_state(self, region_width: Tensor) -> Tensor:
        """Grid voxel width -> this encoder's region-configuration vector.

        Pure and traceable, so it can be exported as its own graph: a deployed consumer runs it
        ONLY when the queried grid resolution changes, caches the result, and hands it back on
        every forward. Region-agnostic encoders return an empty vector.
        """
        return region_width.new_zeros(0)

    def set_query_region(self, region_width: float | None) -> None:
        """Announce the spatial extent each query point REPRESENTS, in the encoder's own input
        frame (e.g. the voxel size of the grid being queried; None = a true point query).

        Region-aware encodings (integrated/anti-aliased ones) use it to prefilter their features;
        plain point encodings ignore it. Callers announce their query geometry through this hook
        instead of the encoders leaking their type into the models. Callers that bypass the
        model's volume assembly (single-voxel / arbitrary-position queries) must call this
        themselves — see ``FeedforwardPointwiseModel.set_query_region``.
        """
        return None

    def forward(self, x: Tensor, region_state: Tensor | None = None) -> Tensor:
        """Encode ``x``. ``region_state`` is the vector from ``compute_region_state`` describing
        the extent each query represents; region-agnostic encodings ignore it."""
        ...
