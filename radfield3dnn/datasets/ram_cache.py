from collections import OrderedDict
import torch
from torch.utils.data import Dataset


def compute_ram_budget(override_gb: float = None, margin_fraction: float = 0.15) -> int:
    if override_gb is not None:
        return int(float(override_gb) * 1e9)
    try:
        import psutil
        vm = psutil.virtual_memory()
        reserve = max(0.0, min(0.5, float(margin_fraction))) * vm.total   # keep ~10–20 % of RAM free
        return int(max(0.0, vm.available - reserve))
    except Exception:
        return int(4e9)   # conservative fallback if psutil is unavailable


def tensor_bytes(obj) -> int:
    """Recursively sum the byte size of every tensor in a (possibly nested namedtuple/list) object."""
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    if hasattr(obj, "_fields"):          # namedtuples
        return sum(tensor_bytes(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    return 0


class RamCachedDataset(Dataset):
    """Elastic, memory-pressure-aware RAM cache decorator for field datasets.

    Self-tunes the cached set during training instead of freezing at a fixed size:

    The pressure signal is primarily **swap growth**, because ``virtual_memory().available`` counts
    reclaimable buff/cache and *lags* — it can read "fine" while the kernel is already swapping. So if
    the training process's footprint rises later, the cache gives RAM back; if memory frees up, the
    cache fills again toward the largest safe size. ``max_bytes`` is an optional soft per-process cap.
    Checks are amortized every ``check_every`` items so psutil cost is negligible.
    """

    def __init__(self, decoratee: Dataset, max_bytes: int = 0, min_free_bytes: int = 0,
                 headroom_bytes: int = 4_000_000_000, swap_grow_bytes: int = 2_000_000_000,
                 check_every: int = 8, evict_per_check: int = 8):
        self.decoratee = decoratee
        self.max_bytes = int(max_bytes) if max_bytes else 0
        self.min_free_bytes = int(min_free_bytes) if min_free_bytes else 0
        self.headroom_bytes = int(headroom_bytes)
        self.swap_grow_bytes = int(swap_grow_bytes)
        self.check_every = max(1, int(check_every))
        self.evict_per_check = max(1, int(evict_per_check))
        self._cache: "OrderedDict" = OrderedDict()
        self._bytes = 0
        self._since_check = 0
        self._base_swap = self._swap_used()

    def __len__(self):
        return len(self.decoratee)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.decoratee, name)

    @staticmethod
    def _swap_used() -> int:
        try:
            import psutil
            return int(psutil.swap_memory().used)
        except Exception:
            return 0

    def _evict(self, n: int):
        for _ in range(n):
            if not self._cache:
                break
            _, old = self._cache.popitem(last=False)   # oldest
            self._bytes -= tensor_bytes(old)

    def __getitem__(self, idx):
        hit = self._cache.get(idx)
        if hit is not None:
            self._cache.move_to_end(idx)               # mark recently used
            return hit

        item = self.decoratee[idx]
        self._since_check += 1
        if self._since_check >= self.check_every:
            self._since_check = 0
            try:
                import psutil
                vm = psutil.virtual_memory()
                swap_grew = (self._swap_used() - self._base_swap) > self.swap_grow_bytes
                budget_ok = (not self.max_bytes) or (self._bytes <= self.max_bytes)
                if (self.min_free_bytes and vm.available < self.min_free_bytes) or swap_grew:
                    self._evict(self.evict_per_check)                       # pressure -> shrink
                elif budget_ok and (not self.min_free_bytes
                                    or vm.available > self.min_free_bytes + self.headroom_bytes):
                    self._cache[idx] = item                                     # comfortable -> grow
                    self._bytes += tensor_bytes(item)
            except Exception:
                pass
        return item

    @property
    def cached_count(self) -> int:
        return len(self._cache)

    @property
    def cached_bytes(self) -> int:
        return self._bytes
