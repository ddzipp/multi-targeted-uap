import argparse
from dataclasses import asdict, dataclass

import yaml


@dataclass
class Config:
    """Config Class
    Support dataclass, config file, and command line arguments
    Returns:
        _type_: dataclass
    """

    # dataset info
    dataset_name: str = "ImageNet"

    # model info
    model_name: str = "resnet50"

    # attack info
    optimizer: str = "momentum"
    mu: float = 0.9
    epoch: int = 500
    lr: float = 0.1
    attack_mode: str = "frame"
    norm_type: str = "linf"
    norm_epsilon: float = 1.0
    frame_width: int = 6
    patch_size: int = 40
    patch_location: tuple = (0, 0)

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
