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

    def deploy_stage(self) -> str | None:
        """The RF3M stage this encoder ships as, or None to be traced into the graph around it.

        Almost every encoder is arithmetic torch.onnx can lower, so it becomes part of whichever
        graph calls it and this returns None. An encoder backed by a CUDA custom op — the
        tiny-cuda-nn hash grid — has no ONNX lowering at all: tracing it either refuses or, worse,
        emits a graph silently missing the encoder. Such an encoder announces a stage name here,
        and the packager ships it as compiled code beside the graphs, with a buffer carrying its
        output to the trunk.

        Callers ask through this hook rather than testing the encoder's type, so adding another
        kernel-backed encoding does not touch the export path.
        """
        return None

    def deploy_stage_blocks(self) -> dict:
        """The compiled forms of this encoder's stage, as {block kind: payload bytes}.

        Called only when `deploy_stage()` names a stage. Each entry becomes a block of the package
        sharing that name, and the runtime picks the one that runs on the device it finds.
        """
        raise NotImplementedError(f"{type(self).__name__} announces no deploy stage")

    def deploy_stage_weights(self):
        """`(payload bytes, element count, dtype name)` for this encoder's stage weights buffer.

        ONE buffer, shared by every compiled form of the stage, so a CUDA path and any future
        path cannot drift apart. Called only when `deploy_stage()` names a stage.
        """
        raise NotImplementedError(f"{type(self).__name__} announces no deploy stage")

    def deploy_stage_launch(self) -> dict:
        """`{block kind: {entry_point, block_size, grid_size, shared_memory_bytes}}` for the
        compiled forms. A blob of machine code cannot say how to call itself."""
        raise NotImplementedError(f"{type(self).__name__} announces no deploy stage")

    def table_parameters(self):
        """Learnable lookup-table parameters (hash-grid / feature-grid entries). They are
        optimised as embeddings — high learning rate, no weight decay — unlike the MLP weights, so
        the optimizer collects them through this hook rather than by attribute name. Parameter-free
        and analytic encodings yield nothing."""
        return iter(())

    def forward(self, x: Tensor, region_state: Tensor | None = None) -> Tensor:
        """Encode ``x``. ``region_state`` is the vector from ``compute_region_state`` describing
        the extent each query represents; region-agnostic encodings ignore it."""
        ...
