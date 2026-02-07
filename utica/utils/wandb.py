# utica/utils/wandb.py
import wandb
from typing import Optional
from omegaconf import OmegaConf

def wandb_init(cfg, is_main_process: bool = True) -> Optional["wandb.sdk.wandb_run.Run"]:
    # Convert OmegaConf -> dict (resolves interpolations)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg

    wandb_cfg = (cfg_dict.get("wandb") or {})
    if not wandb_cfg.get("enabled", False):
        return None
    if not is_main_process:
        return None

    run = wandb.init(
        project=wandb_cfg.get("project", "debug"),
        name=wandb_cfg.get("name", None),
        config=cfg_dict,          # <-- logs full cfg under run.config
        # optionally:
        save_code=True,
        # settings=wandb.Settings(start_method="thread"),
    )
    return run

def wandb_log(run, payload: dict, step: int):
    if run is None:
        return
    run.log(payload, step=step)   # use run.log to avoid any global state surprises

def wandb_finish(run):
    if run is None:
        return
    run.finish()
