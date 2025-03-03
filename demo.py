import os

import torch
from torch.utils.data import Subset
from tqdm import tqdm

from attacks.base import Attacker
from config.config import Config
from dataset import collate_fn, load_dataset
from dataset.base import VisionData
from models import get_model
from utils.constraint import Constraint
from utils.logger import WBLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
torch.manual_seed(42)


def attack_dataloader(dataset_name: str, transform=None, target=None):
    # Set multi-target labels
    target0, target1 = "WARNING!", "ERROR!"
    # target0, target1 = 464, 752
    sample_id = torch.tensor(list(range(0, 30)) + list(range(1000, 1030)))
    dataset = load_dataset(dataset_name, target=target0, transform=transform)
    dataset_0 = Subset(dataset, sample_id[:30])
    dataset = load_dataset(dataset_name, target=target1, transform=transform)
    dataset_1 = Subset(dataset, sample_id[30:])
    dataset = torch.utils.data.ConcatDataset([dataset_0, dataset_1])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return dataloader


def main():
    # init
    cfg = Config()
    model = get_model(cfg.model_name)
    dataloader = attack_dataloader(cfg.dataset_name)
    constraint = Constraint(cfg.attack_mode, frame_width=cfg.frame_width)
    attacker = Attacker(model, constraint, cfg)
    run = WBLogger(
        project="multi-targeted-VLM-test",
        config=cfg,
        name="perturbation-on-01-image",
    ).run
    # accelerator = Accelerator()
    # model, dataloader = accelerator.prepare(model, dataloader)
    try:
        # train loop
        with tqdm(range(cfg.epoch)) as pbar:
            for _ in pbar:
                loss = attacker.trainer(dataloader)
                run.log({"loss": loss})
                pbar.set_postfix({"loss": f"{loss:.2f}"})
    finally:
        # save perturbation and mask
        attacker.saver(filename := "./save/perturbation.pth")
        run.save(filename, base_path="save")


if __name__ == "__main__":
    main()
