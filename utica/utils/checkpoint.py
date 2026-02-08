from pathlib import Path
import torch

from utica.utils.ddp import is_main_process


def save_checkpoint(ckpt_dir: Path, iteration: int, model, optimizer):
    if not is_main_process():
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / f"{iteration}.pth"

    m = model.module if hasattr(model, "module") else model

    payload = {
        "iteration": iteration,
        "student": m.student.state_dict(),
        "teacher": m.teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, path)
