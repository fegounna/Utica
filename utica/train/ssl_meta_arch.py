import gc
from functools import partial

import torch
from torch import Tensor, nn

from utica.layers.dino_head import DINOHead

from utica.model import build_model_from_cfg

from utica.loss.dino_clstoken_loss import DINOLoss
from utica.loss.koleo_loss import KoLeoLoss
from utica.loss.ibot_patch_loss import iBOTPatchLoss

from utica.utils.param_groups import get_params_groups_with_decay, fuse_params_groups


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        torch.cuda.empty_cache()
        gc.collect()

        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        self.embed_dim = embed_dim  # D
        self.dino_out_dim = cfg.dino.head_n_prototypes  # K

        dino_head_class = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )
        student_model_dict["dino_head"] = dino_head_class()
        teacher_model_dict["dino_head"] = dino_head_class()
        self.dino_loss = DINOLoss(self.dino_out_dim)

        self.koleo_loss = KoLeoLoss()

        ibot_head_class = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.ibot.head_n_prototypes,
            hidden_dim=cfg.ibot.head_hidden_dim,
            bottleneck_dim=cfg.ibot.head_bottleneck_dim,
            nlayers=cfg.ibot.head_nlayers,
        )
        student_model_dict["ibot_head"] = ibot_head_class()
        teacher_model_dict["ibot_head"] = ibot_head_class()
        self.ibot_patch_loss = iBOTPatchLoss(cfg.ibot.head_n_prototypes)

        # Build student and teacher models
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        self.teacher.requires_grad_(False)

        self.dino_global_ignore_diagonal = self.cfg.dino.global_ignore_diagonal
        self.dino_loss_weight = self.cfg.dino.loss_weight
        self.dino_koleo_loss_weight = self.cfg.dino.koleo_loss_weight
        self.ibot_loss_weight = self.cfg.ibot.loss_weight

    def init_weights(self) -> None:
        self.student.dino_head.init_weights()
        self.student.ibot_head.init_weights()
        self.dino_loss.init_weights()
        self.ibot_patch_loss.init_weights()

        self.teacher.load_state_dict(self.student.state_dict(), strict=True)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def forward_backward(
        self, data, teacher_temp
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        n_global_crops = 2
        n_local_crops = self.cfg.crops.local_crops_number
        B = data["collated_local_crops"].shape[0] // n_local_crops

        global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        local_crops = data["collated_local_crops"].cuda(non_blocking=True)

        masks = data["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = data["mask_indices_list"].cuda(non_blocking=True)
        masks_weight = data["masks_weight"].cuda(non_blocking=True)
        n_masked_patches_tensor = data["n_masked_patches"].cuda(non_blocking=True)

        teacher_global = self.get_teacher_output(
            global_crops.unflatten(0, (n_global_crops, B)),
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
        )
        student_global, student_local = self.get_student_output(
            global_crops=global_crops.unflatten(0, (n_global_crops, B)),
            local_crops=local_crops.unflatten(0, (n_local_crops, B)),
            masks=masks,
            mask_indices_list=mask_indices_list,
        )

        loss_accumulator, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
        )

        self.backprop_loss(loss_accumulator)

        return loss_accumulator, loss_dict

    @torch.no_grad()
    def get_teacher_output(
        self,
        time_series,
        mask_indices_list,
        teacher_temp,
        n_masked_patches_tensor,
    ):
        n_crops, B, T = time_series.shape
        time_series = time_series.flatten(0, 1)  # n_crops*B,T

        backbone_out = self.teacher.backbone(time_series, is_training=True) 
        cls = backbone_out["x_norm_clstoken"]  # n_crops * B, D
        ibot_patch = backbone_out["x_norm_patchtokens"]  # n_crops * B, P, D

        # IBOT head only on patches that are masked for the student
        buffer = torch.index_select(
            ibot_patch.flatten(0, 1), dim=0, index=mask_indices_list
        )
        masked_patch_after_head = self.teacher.ibot_head(buffer)

        # DINO head on CLS tokens
        cls_after_head = self.teacher.dino_head(cls)  # [n_crops * B, K]

        # Center with sinkhorn-knopp
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
            cls_after_head, teacher_temp=teacher_temp
        )  # [n_crops * B, K]
        cls_centered = cls_centered.unflatten(0, (n_crops, B))  # [n_crops, B, K]
        masked_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
            masked_patch_after_head,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
        )  # [n_masked_patches, K]

        return {
            "cls_pre_head": cls.unflatten(0, [n_crops, B]),  # [n_crops, B, D]
            "patch_pre_head": ibot_patch.unflatten(
                0, [n_crops, B]
            ),  # [n_crops, B, P, D]
            "cls_after_head": cls_after_head.unflatten(
                0, [n_crops, B]
            ),  # [n_crops, B, K]
            "cls_centered": cls_centered,  # [n_crops, B, K]
            "masked_patch_centered": masked_patch_centered,  # [n_masked_patches, K]
        }

    def get_student_output(self, global_crops, local_crops, masks, mask_indices_list):
        n_global_crops, B, T = global_crops.shape
        n_local_crops, B, T = local_crops.shape

        global_crops = global_crops.flatten(0, 1)

        global_out = self.student["backbone"](
            global_crops, masks=masks, is_training=True
        )
        local_out = self.student["backbone"](
            local_crops.flatten(0, 1), masks=None, is_training=True
        )

        g_cls, g_patch = (
            global_out["x_norm_clstoken"],
            global_out["x_norm_patchtokens"],
        )
        l_cls, l_patch = (
            local_out["x_norm_clstoken"],
            local_out["x_norm_patchtokens"],
        )

        # IBOT head only on masked patches
        masked_patches_pre_head = torch.index_select(
            g_patch.flatten(0, 1), dim=0, index=mask_indices_list
        )
        global_masked_patch_after_head = self.student.ibot_head(masked_patches_pre_head)

        # DINO head on CLS tokens (all in one pass)
        buffer = [
            g_cls,  # [n_global_crops * B, D]
            l_cls,  # [n_local_crops * B, D]
        ]
        sizes = [x.shape[0] for x in buffer]
        buffer = torch.cat(buffer, dim=0)  # [n_global_crops * B + n_local_crops * B, D]
        buffer = self.student.dino_head(
            buffer
        )  # [n_global_crops * B + n_local_crops * B, K]
        buffer = torch.split_with_sizes(buffer, sizes, dim=0)

        global_out = {
            "cls_pre_head": g_cls.unflatten(
                0, [n_global_crops, B]
            ),  # [n_global_crops, B, D]
            "patch_pre_head": g_patch.unflatten(
                0, [n_global_crops, B]
            ),  # [n_global_crops, B, P, D]
            "cls_after_head": buffer[0].unflatten(
                0, [n_global_crops, B]
            ),  # [n_global_crops, B, K],
            "masked_patch_after_head": global_masked_patch_after_head,  # [n_masked_patches, K]
            "masked_patch_pre_head": masked_patches_pre_head,  # [n_masked_patches, D]
        }
        local_out = {
            "cls_pre_head": l_cls.unflatten(
                0, [n_local_crops, B]
            ),  # [n_local_crops, B, D]
            "patch_pre_head": l_patch.unflatten(
                0, [n_local_crops, B]
            ),  # [n_local_crops, B, P, D]
            "cls_after_head": buffer[1].unflatten(
                0, [n_local_crops, B]
            ),  # [n_local_crops, B, K],
        }

        return global_out, local_out

    def compute_losses(
        self,
        teacher_global,
        student_global,
        student_local,
        masks,
        mask_indices_list,
        masks_weight,
    ):
        n_global_crops = student_global["cls_after_head"].shape[0]
        n_local_crops = student_local["cls_after_head"].shape[0]
        loss_dict = {}
        loss_accumulator = 0.0

        # Loss scales like in DINOv2, these are multiplied with the loss weights from the config
        dino_global_terms = (
            n_global_crops * (n_global_crops - 1)
            if self.dino_global_ignore_diagonal
            else n_global_crops**2
        )
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
        koleo_scale = n_global_crops

        # DINO local loss: compare post-head CLS tokens: student(local crops) vs. teacher(global crops)
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        loss_accumulator += (
            self.dino_loss_weight * dino_local_scale * dino_local_crops_loss
        )

        # DINO global loss: compare post-head CLS tokens: student(global crops) vs. teacher(global crops)
        dino_global_crops_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.dino_global_ignore_diagonal,
        )
        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
        loss_accumulator += (
            self.dino_loss_weight * dino_global_scale * dino_global_crops_loss
        )

        # Koleo: regularize pre-head CLS tokens of student(global crops)
        koleo_loss = (
            sum(self.koleo_loss(x) for x in student_global["cls_pre_head"])
            / n_global_crops
        )
        loss_dict["koleo_loss"] = koleo_loss
        loss_accumulator += self.dino_koleo_loss_weight * koleo_scale * koleo_loss

        # IBOT loss
        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
            student_global["masked_patch_after_head"],
            teacher_global["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=mask_indices_list.shape[0],
            masks_weight=masks_weight,
        )
        loss_dict["ibot_loss"] = ibot_patch_loss
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        return loss_accumulator, loss_dict

    def backprop_loss(self, loss):
        loss.backward()

    @torch.no_grad()
    def update_teacher(self, m: float):
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.mul_(m).add_(ps, alpha=1.0 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            print(m)
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups
