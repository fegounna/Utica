from pathlib import Path
import torch
from utica import distributed as distu


def save_checkpoint_ddp(
    ckpt_dir: Path, iteration: int, model, optimizer, overwrite=True
):
    if not distu.is_main_process():
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / f"{iteration}.pth"
    if path.exists() and not overwrite:
        return

    # unwrap DDP if needed
    m = model.module if hasattr(model, "module") else model

    payload = {
        "iteration": iteration,
        "student": m.student.state_dict(),
        "teacher": m.teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        # optionally: "cfg": m.cfg,
    }
    torch.save(payload, path)
