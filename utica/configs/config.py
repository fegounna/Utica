from omegaconf import OmegaConf
from pathlib import Path


def setup_config():
    cfg_path = Path(__file__).with_name("config.yaml")
    return OmegaConf.load(cfg_path)
