"""Adapted for mantis names"""

from collections import defaultdict


def get_vit_lr_decay_rate(
    name,
    lr_decay_rate,
    num_layers,
):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("vit_unit"):
        if ".mask_token" in name or ".cls_token" in name:
            layer_id = 0
        elif ".layers." in name:
            layer_id = int(name[name.find(".layers.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_params_groups_with_decay(model, lr_decay_rate, patch_embed_lr_mult):
    if hasattr(model, "n_blocks"):
        n_blocks = model.n_blocks
    else:
        n_blocks = 0

    all_param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(name, lr_decay_rate, num_layers=n_blocks)
        d = {
            "params": param,
            "is_last_layer": False,
            "lr_multiplier": decay_rate,
            "wd_multiplier": 1.0,
            "name": name,
        }

        if "last_layer" in name:
            d.update({"is_last_layer": True})

        if name.endswith(".bias") or "norm" in name or "gamma" in name:
            d.update({"wd_multiplier": 0.0})

        if "tokgen_unit" in name:
            d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})

        all_param_groups.append(d)

        print(
            f"""{name}: lr_multiplier: {d["lr_multiplier"]}, wd_multiplier: {d["wd_multiplier"]}"""
        )
    return all_param_groups


def fuse_params_groups(
    all_params_groups, keys=("lr_multiplier", "wd_multiplier", "is_last_layer")
):
    fused_params_groups = defaultdict(lambda: {"params": []})
    for d in all_params_groups:
        identifier = ""
        for k in keys:
            identifier += k + str(d[k]) + "_"

        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]["params"].append(d["params"])

    return fused_params_groups.values()
