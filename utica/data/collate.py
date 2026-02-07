import random

import torch


def collate_data(
    samples_list,
    mask_generator,
    dtype=torch.float32,
    n_tokens=32,
    mask_ratio_tuple=(0.1, 0.6),
    mask_probability=0.5,
):
    n_global_crops = len(samples_list[0]["global_crops"])
    n_local_crops = len(samples_list[0]["local_crops"])

    collated_global_crops = torch.stack(
        [s["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    )  # 2 global crops
    collated_local_crops = torch.stack(
        [s["local_crops"][i] for i in range(n_local_crops) for s in samples_list]
    )

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)

    probs = torch.linspace(
        *mask_ratio_tuple, n_samples_masked + 1
    )  # schedule of mask ratios

    masks_list = []
    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        masks_list.append(mask)
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (
        (1 / collated_masks.sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(collated_masks)[collated_masks]
    )

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "n_masked_patches": torch.full(
            (1,), fill_value=mask_indices_list.shape[0], dtype=torch.long
        ),
    }
    return out
