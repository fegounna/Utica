from pathlib import Path
from typing import List, Callable

import numpy as np
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """
    Each file: (N_i, C_i, 512)
    """

    def __init__(
        self,
        npy_paths: List[Path],
        transforms: Callable | None = None,
        t_dim: int = 512,
        cast_dtype=np.float32,
    ):
        self.t_dim = t_dim
        self.cast_dtype = cast_dtype
        self.transforms = transforms

        self.mmaps = []
        for p in npy_paths:
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 3 or arr.shape[2] != t_dim:
                raise ValueError(f"{p} expected (*, *, {t_dim}), got {arr.shape}")
            self.mmaps.append(arr.reshape(-1, t_dim))

        counts = [m.shape[0] for m in self.mmaps]
        self.cumsum = np.cumsum([0] + counts)
        self.total_len = self.cumsum[-1]

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int):
        file_idx = np.searchsorted(self.cumsum[1:], idx, side="right")
        local_idx = idx - self.cumsum[file_idx]

        x = self.mmaps[file_idx][local_idx]

        if self.cast_dtype is not None:
            x = x.astype(self.cast_dtype, copy=False)

        if self.transforms is not None:
            x = self.transforms(x)
        return x
