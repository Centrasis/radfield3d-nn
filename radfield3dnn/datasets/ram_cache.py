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
    """Fill-then-freeze RAM cache decorator for field datasets.

    Caches as many raw decoded fields as safely fit, then FREEZES and serves the cached ones while
    streaming the overflow from disk. Filling is monotonic — never evict-and-refetch — so it never
    oscillates or evicts the OS page cache (which keeps the streamed .rf3 reads fast); on a dataset
    larger than RAM it simply caches the part that fits. Stops growing at ``max_bytes`` (hard
    per-process cap) or once live free RAM would drop below ``min_free_bytes``, whichever comes first.
    """

    def __init__(self, decoratee: Dataset, max_bytes: int = 0, min_free_bytes: int = 0,
                 check_every: int = 8):
        self.decoratee = decoratee
        self.max_bytes = int(max_bytes) if max_bytes else 0
        self.min_free_bytes = int(min_free_bytes) if min_free_bytes else 0
        self.check_every = max(1, int(check_every))
        self._cache: dict = {}
        self._bytes = 0
        self._since_check = 0
        self._frozen = False

    def __len__(self):
        return len(self.decoratee)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.decoratee, name)

    def _ram_pressure(self) -> bool:
        if not self.min_free_bytes:
            return False
        try:
            import psutil
            return psutil.virtual_memory().available < self.min_free_bytes
        except Exception:
            return False

    def preload(self, verbose: bool = True):
        """Fill the cache in the CURRENT (parent) process up to the budget, then freeze. The
        DataLoader workers are forked AFTER this, so they all inherit this one set copy-on-write —
        a single shared pool (each field decoded/stored once, every worker serves from it) instead
        of each worker building its own. Monotonic; stops at the byte cap / RAM floor."""
        n = len(self.decoratee)
        from rich import print as rprint
        for i, idx in enumerate(range(n)):
            if self.max_bytes and self._bytes >= self.max_bytes:
                break
            if (i % self.check_every == 0) and self._ram_pressure():
                break
            item = self.decoratee[idx]
            self._cache[idx] = item
            self._bytes += tensor_bytes(item)
            if verbose and i and i % 256 == 0:
                rprint(f"[blue]  RAM cache preloading… {i}/{n} fields, {self._bytes/1e9:.1f} GB[/blue]")
        self._frozen = True   # workers inherit this -> they never add private copies, only stream overflow
        if verbose:
            from rich import print as rprint
            rprint(f"[green]RAM cache: preloaded {len(self._cache)}/{n} fields "
                   f"({self._bytes/1e9:.1f} GB) in the parent; workers share it copy-on-write, "
                   f"the rest streams from disk.[/green]")
        return self

    def __getitem__(self, idx):
        hit = self._cache.get(idx)
        if hit is not None:
            return hit

        item = self.decoratee[idx]
        if self._frozen:
            return item

        if self.max_bytes and self._bytes >= self.max_bytes:
            self._frozen = True            # hit the byte cap -> stop growing
            return item
        self._since_check += 1
        if self._since_check >= self.check_every:
            self._since_check = 0
            if self._ram_pressure():
                self._frozen = True        # hit the RAM floor -> stop growing, stream the rest
                return item

        self._cache[idx] = item
        self._bytes += tensor_bytes(item)
        return item

    @property
    def cached_count(self) -> int:
        return len(self._cache)

    @property
    def cached_bytes(self) -> int:
        return self._bytes
