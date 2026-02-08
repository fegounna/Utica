import torch.distributed as dist


def is_distributed_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_distributed():
    if dist.is_initialized():
        return
    dist.init_process_group(backend="nccl")


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process():
    return get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def get_process_subgroup():
    if dist.is_initialized():
        return dist.group.WORLD
    return None


def get_subgroup_size():
    return get_world_size()


def all_reduce_mean(tensor):
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor
