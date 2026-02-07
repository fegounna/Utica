import torch
from utica.utils.ddp import all_reduce_mean

def _prob_stats(probs: torch.Tensor, prefix: str) -> dict:
    """
    probs: [..., K] on GPU
    returns dict of scalar tensors
    """
    eps = 1e-6
    p = probs.clamp(min=eps, max=1.0)

    ent = -(p * p.log()).sum(dim=-1).mean()
    pmax = p.max(dim=-1).values.mean()

    # mean assignment over batch
    q = p.mean(dim=tuple(range(p.ndim - 1)))  # [K]
    q = q.clamp(min=eps)
    q_ent = -(q * q.log()).sum()

    return {
        f"{prefix}/entropy": ent,
        f"{prefix}/pmax": pmax,
        f"{prefix}/mean_assign_entropy": q_ent,
        f"{prefix}/mean_assign_min": q.min(),
        f"{prefix}/mean_assign_max": q.max(),
    }


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

