from .architecture import Mantis8M


def build_model_from_cfg(cfg):
    student = Mantis8M()
    teacher = Mantis8M()
    embed_dim = student.hidden_dim
    return student, teacher, embed_dim
