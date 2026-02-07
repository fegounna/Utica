import numpy as np
import torch
from functools import partial


class DataAugmentation:
    def __init__(
        self,
        global_crop_removal_range,  # (min_crop_rate, max_crop_rate)
        local_crop_removal_range,  # (min_crop_rate, max_crop_rate)
        local_crops_number,
        global_crops_size=512,
        local_crops_size=256,
        jitter_sigma=0.2,
        seed=None,
    ):
        self.local_crops_number = local_crops_number
        self.jitter_sigma = float(jitter_sigma)

        self.rng = np.random.default_rng(seed)

        self.global_crop = partial(
            self.random_crop_resize,
            crop_rate_range=global_crop_removal_range,
            size=global_crops_size,
        )
        self.local_crop = partial(
            self.random_crop_resize,
            crop_rate_range=local_crop_removal_range,
            size=local_crops_size,
        )

    def random_crop_resize(self, x, crop_rate_range: tuple, size=None):
        crop_rate = self.rng.uniform(crop_rate_range[0], crop_rate_range[1])

        seq_len = int(x.shape[-1])
        size = seq_len if size is None else int(size)

        cropped_seq_len = int(seq_len * (1 - crop_rate))

        start_idx = self.rng.integers(0, seq_len - cropped_seq_len + 1)
        x_cropped = x[start_idx : start_idx + cropped_seq_len]

        old_idx = np.linspace(0.0, 1.0, num=cropped_seq_len, endpoint=True)
        new_idx = np.linspace(0.0, 1.0, num=size, endpoint=True)
        return np.interp(new_idx, old_idx, x_cropped)

    def jitter(self, x):
        scale = float(np.std(x))
        noise_std = self.jitter_sigma * scale
        return x + self.rng.normal(0.0, noise_std, size=x.shape)

    def maybe_jitter(self, x, p: float):
        if self.rng.random() < p:
            return self.jitter(x)
        return x

    def __call__(self, x):
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D input (T,), got shape {x.shape}")

        # global crops
        g1 = self.maybe_jitter(self.global_crop(x), p=1.0)
        g2 = self.maybe_jitter(self.global_crop(x), p=0.1)

        # local crops
        locals_ = [
            self.maybe_jitter(self.local_crop(x), p=0.5)
            for _ in range(self.local_crops_number)
        ]

        g1 = torch.from_numpy(g1).float()
        g2 = torch.from_numpy(g2).float()
        locals_ = [torch.from_numpy(a).float() for a in locals_]

        return {"global_crops": [g1, g2], "local_crops": locals_}
