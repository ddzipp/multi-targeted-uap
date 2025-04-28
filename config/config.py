import argparse
from dataclasses import asdict, dataclass

import yaml  # type: ignore


@dataclass
class Config:
    """Config Class
    Support dataclass, config file, and command line arguments
    Returns:
        _type_: dataclass
    """

    # dataset info
    # dataset_name: str = "VQA"
    dataset_name: str = "ImageNet"
    split: str = "train"
    batch_size: int = 4
    num_targets: int = 3
    train_size: int = 50
    targets: dict | None = None
    sample_id: list | None = None
    # model info
    model_name: str = "Qwen2_5"  # renset50, Llava, Qwen2_5

    # attack info
    attack_name: str = "base"  # base, split, union_split
    lr: float = 0.01
    epoch: int = 500
    attack_mode: str = "frame" # frame, corner, pixel
    bound: tuple = (0, 1)
    epsilon: tuple = (-32 / 255, 32 / 255)
    frame_width: int = 6
    patch_size: tuple[int, int] = (20, 20)
    patch_location: tuple[int, int] = (0, 0)
    on_normalized: bool = True
    save_dir: str = f"./save/VLM/Margin/{attack_mode}/{model_name}_T{num_targets}"

    def asdict(self):
        return asdict(self)

    def __post_init__(self):
        # command line arguments
        parser = argparse.ArgumentParser(description="Config file path")
        parser.add_argument(
            "-f",
            "--config_file",
            type=str,
            default=None,
            help="Path to the config file",
        )
        args, unknown_args = parser.parse_known_args()
        self.read_config_file(args.config_file)
        # update config with all command line arguments
        for key, value in vars(args).items():
            if value:
                setattr(self, key, value)

    def read_config_file(self, file_path=None):
        if file_path is None:
            return
        with open(file_path, "r", encoding="utf-8") as file:
            cfg_file = yaml.safe_load(file)
        # update config with config file
        for key, value in cfg_file.items():
            setattr(self, key, value)
