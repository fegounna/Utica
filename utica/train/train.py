import gc
import os
from functools import partial
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from utica.train.ssl_meta_arch import SSLMetaArch
from utica.train.cosine_lr_scheduler import CosineScheduler

from utica.data.augmentations import DataAugmentation
from utica.data.dataset import SyntheticDataset
from utica.data.masking import MaskingGenerator1D
from utica.data.collate import collate_data

from utica.configs import setup_config

from utica.utils.ddp import (
    init_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
)
from utica.utils.wandb import wandb_init, wandb_log, wandb_finish
from utica.utils.debug import reduce_metrics_mean
from utica.utils.checkpoint import save_checkpoint


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(
        params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2)
    )


def build_schedulers(cfg, iters_per_epoch: int):
    # iters_per_epoch = int(cfg.optim.epochs) * iters_per_epoch
    total_iters = int(cfg.optim.epochs) * iters_per_epoch

    lr = dict(
        base_value=float(cfg.optim.lr),
        final_value=float(cfg.optim.min_lr),
        total_iters=total_iters,
        warmup_iters=int(cfg.optim.warmup_epochs * iters_per_epoch),
        start_warmup_value=0.0,
        trunc_extra=float(getattr(cfg.optim, "schedule_trunc_extra", 0.0)),
    )
    wd = dict(
        base_value=float(cfg.optim.weight_decay),
        final_value=float(cfg.optim.weight_decay_end),
        total_iters=total_iters,
        trunc_extra=float(getattr(cfg.optim, "schedule_trunc_extra", 0.0)),
    )
    momentum = dict(
        base_value=float(cfg.teacher.momentum_teacher),
        final_value=float(cfg.teacher.final_momentum_teacher),
        total_iters=total_iters,
        trunc_extra=float(getattr(cfg.optim, "schedule_trunc_extra", 0.0)),
    )
    teacher_temp = dict(
        base_value=float(cfg.teacher.teacher_temp),
        final_value=float(cfg.teacher.teacher_temp),
        total_iters=int(cfg.teacher.warmup_teacher_temp_epochs * iters_per_epoch),
        warmup_iters=int(cfg.teacher.warmup_teacher_temp_epochs * iters_per_epoch),
        start_warmup_value=float(cfg.teacher.warmup_teacher_temp),
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    freeze = int(cfg.optim.freeze_last_layer_epochs * iters_per_epoch)
    last_layer_lr_schedule.schedule[:freeze] = 0.0

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for pg in optimizer.param_groups:
        is_last_layer = pg["is_last_layer"]
        lr_multiplier = pg["lr_multiplier"]
        wd_multiplier = pg["wd_multiplier"]

        pg["weight_decay"] = float(wd) * float(wd_multiplier)
        if is_last_layer:
            pg["lr"] = float(last_layer_lr) * float(lr_multiplier)
        else:
            pg["lr"] = float(lr) * float(lr_multiplier)


def build_data_loader_from_cfg(cfg):
    aug = DataAugmentation(
        global_crop_rate_range=tuple(cfg.crops.global_crop_rate_range),
        local_crop_rate_range=tuple(cfg.crops.local_crop_rate_range),
        local_crops_number=int(cfg.crops.local_crops_number),
        global_crops_size=int(cfg.crops.global_crops_size),
        local_crops_size=int(cfg.crops.local_crops_size),
        jitter_sigma=tuple(cfg.crops.jitter_sigma),
        seed=int(cfg.train.seed),
    )

    dataset = SyntheticDataset(
        npy_paths=[Path(p) for p in cfg.train.npy_paths],
        transforms=aug,
        t_dim=int(cfg.crops.global_crops_size),
    )
    print(f"Dataset length: {len(dataset)}")

    mask_gen = MaskingGenerator1D(
        num_patches=int(cfg.model.num_patches), seed=int(cfg.train.seed)
    )

    collate_fn = partial(
        collate_data,
        mask_generator=mask_gen,
        dtype=torch.float32,
        n_tokens=int(cfg.model.num_patches),
        mask_ratio_tuple=tuple(cfg.ibot.mask_ratio_min_max),
        mask_probability=float(cfg.ibot.mask_sample_probability),
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
        seed=int(cfg.train.seed),
        drop_last=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size_per_gpu),
        sampler=sampler,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=(int(cfg.train.num_workers) > 0),
    )

    return loader, sampler


def do_train(cfg, model: SSLMetaArch, run=None):
    ckpt_dir = Path(cfg.train.output_dir, "ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_loader, sampler = build_data_loader_from_cfg(cfg)
    iters_per_epoch = len(data_loader)
    print(f"iters_per_epoch{iters_per_epoch}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model.to(device)
    model.train()

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=bool(getattr(cfg.ddp, "find_unused_parameters", False)),
    )

    optimizer = build_optimizer(cfg, ddp_model.module.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg, iters_per_epoch)

    ddp_model.module.init_weights()

    gc.disable()
    gc.collect()

    iteration = 0

    for epoch in range(int(cfg.optim.epochs)):
        sampler.set_epoch(epoch)

        for batch in data_loader:
            it = iteration

            if (it + 1) % int(getattr(cfg.train, "gc_period", 150)) == 0:
                gc.collect()

            # schedules
            lr = lr_schedule[it]
            wd = wd_schedule[it]
            mom = momentum_schedule[it]
            teacher_temp = teacher_temp_schedule[it]
            last_layer_lr = last_layer_lr_schedule[it]
            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

            optimizer.zero_grad(set_to_none=True)

            total_loss, metrics_dict = ddp_model.module.forward_backward(
                batch,
                teacher_temp=teacher_temp,
            )

            metrics_dict["optim/lr"] = lr
            metrics_dict["optim/wd"] = wd
            metrics_dict["optim/mom"] = mom
            metrics_dict["optim/last_layer_lr"] = last_layer_lr
            metrics_dict["optim/teacher_temp"] = teacher_temp

            clip = cfg.optim.clip_grad
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    ddp_model.module.student.parameters(), max_norm=clip
                )

            with torch.no_grad():
                total_norm_sq = torch.zeros([], device=device)
                for p in ddp_model.module.student.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach().float()
                    total_norm_sq += g.norm() ** 2
                metrics_dict["debug/grad_norm"] = torch.sqrt(total_norm_sq)

            # ---- reduce all metrics across ranks (mean) ----
            metrics_dict["loss/total"] = total_loss.detach()
            metrics_dict = reduce_metrics_mean(metrics_dict, device=device)

            if is_main_process():
                payload = {k: float(v.item()) for k, v in metrics_dict.items()}
                payload["epoch_progress"] = (it + 1) / iters_per_epoch
                wandb_log(run, payload, step=it)

            optimizer.step()

            ddp_model.module.update_teacher(float(mom))

            if (it + 1) % int(cfg.checkpointing.period) == 0:
                torch.cuda.synchronize()
                barrier()
                save_checkpoint(ckpt_dir, it, ddp_model, optimizer)
                barrier()

            iteration += 1


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    init_distributed()
    cfg = setup_config()
    set_seed(int(cfg.train.seed))

    model = SSLMetaArch(cfg)

    run = wandb_init(cfg, is_main_process=is_main_process())
    do_train(cfg, model, run)

    if is_main_process():
        wandb_finish(run)


if __name__ == "__main__":
    main()
