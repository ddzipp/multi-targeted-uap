from attacks.base import Attacker
from attacks.split import SplitAttacker, SplitConstraint
from attacks.union_split import UnionSplitAttacker, UnionSplitConstraint
from config.config import Config
from utils.constraint import Constraint


def get_attacker(cfg: Config, model) -> Attacker:
    if cfg.attack_name == "base":
        constraint = Constraint(
            cfg.attack_mode,
            frame_width=cfg.frame_width,
            patch_size=cfg.patch_size,
            ref_size=299,
        )
        attacker = Attacker(model, constraint, cfg.lr, cfg.on_normalized, bound=cfg.bound, epsilon=cfg.epsilon)
    elif cfg.attack_name == "split":
        constraint = SplitConstraint(
            mode=cfg.attack_mode,
            frame_width=cfg.frame_width,
            patch_size=cfg.patch_size,
            ref_size=299,
            num_targets=len(cfg.targets),
        )
        attacker = SplitAttacker(model, constraint, cfg.lr, cfg.on_normalized, bound=cfg.bound, epsilon=cfg.epsilon)
    elif cfg.attack_name == "union_split":
        constraint = UnionSplitConstraint(
            mode=cfg.attack_mode,
            frame_width=cfg.frame_width,
            patch_size=cfg.patch_size,
            ref_size=299,
            num_targets=len(cfg.targets),
        )
        attacker = UnionSplitAttacker(model, constraint, cfg.lr, cfg.on_normalized, bound=cfg.bound, epsilon=cfg.epsilon)
    else:
        raise ValueError(f"Attack name {cfg.attack_name} not supported")
    return attacker
