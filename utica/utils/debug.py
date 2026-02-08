import torch
from utica.utils.ddp import all_reduce_mean


def _to_scalar_tensor(v, device):
    """Convert v to a 1-element float tensor on device."""
    if torch.is_tensor(v):
        t = v.detach()
        if t.numel() != 1:
            t = t.float().mean()
        else:
            t = t.float()
        return t.to(device)
    return torch.tensor(float(v), device=device, dtype=torch.float32)


def reduce_metrics_mean(metrics: dict, device: torch.device) -> dict:
    out = {}
    for k, v in metrics.items():
        t = _to_scalar_tensor(v, device).clone()
        all_reduce_mean(t)
        out[k] = t
    return out
