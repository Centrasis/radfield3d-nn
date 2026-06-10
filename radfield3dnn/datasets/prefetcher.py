from typing import Optional
import torch
from torch import Tensor
from lightning_utilities.core.apply_func import apply_to_collection


class CudaStreamPrefetcher:
    """
    Prefetches fields non_blocking to GPU memory to overlap training and upload.
    """

    def __init__(self, loader, device: torch.device, processings: Optional[list] = None):
        self.loader = loader
        self.device = device
        self.processings = processings if processings is not None else []
        self._stream = torch.cuda.Stream(device)
        self._uploaded_processings = False
        self._it = None
        self._next = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self._it = iter(self.loader)
        self._preload()
        return self

    def _move(self, batch):
        # apply_to_collection preserves the nested TrainingInputData / PositionalInput /
        # RadiationField namedtuples while moving every Tensor leaf.
        return apply_to_collection(batch, Tensor, lambda t: t.to(self.device, non_blocking=True))

    def _preload(self):
        try:
            raw = next(self._it)
        except StopIteration:
            self._next = None
            return
        with torch.cuda.stream(self._stream):
            batch = self._move(raw)
            if not self._uploaded_processings:
                for i in range(len(self.processings)):
                    self.processings[i] = self.processings[i].to(self.device)
                self._uploaded_processings = True
            # Each processing self-gates on its own training/epoch state, exactly as in the
            # synchronous on_after_batch_transfer path.
            for process in self.processings:
                batch = process(batch)
            self._next = batch

    def __next__(self):
        if self._next is None:
            raise StopIteration
        # Make sure this batch's side-stream upload+preprocess has completed before the main
        # stream consumes it.
        torch.cuda.current_stream().wait_stream(self._stream)
        batch = self._next
        # Tell the allocator the main stream now uses these tensors, so the buffers the side
        # stream produced are not freed/reused until the main stream is done with them.
        cur = torch.cuda.current_stream()
        apply_to_collection(batch, Tensor, lambda t: t.record_stream(cur))
        # Launch the next batch's upload+preprocess on the side stream — this is the overlap with
        # the training_step the caller is about to run on the main stream.
        self._preload()
        return batch
