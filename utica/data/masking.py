import numpy as np


class MaskingGenerator1D:
    def __init__(self, num_patches: int, seed=None):
        self.num_patches = num_patches
        self.rng = np.random.default_rng(seed)

    def __call__(self, num_masking_patches: int):
        mask = np.zeros(self.num_patches, dtype=bool)
        idx = self.rng.choice(self.num_patches, size=num_masking_patches, replace=False)
        mask[idx] = True
        return mask
