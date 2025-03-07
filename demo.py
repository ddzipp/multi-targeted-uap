import os

import torch
from torch.utils.data import Subset
from tqdm import tqdm

from attacks.base import Attacker
from config import Config
from dataset import collate_fn, load_dataset
from dataset.base import VisionData
from models import get_model
from utils.constraint import Constraint
from utils.logger import WBLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
torch.manual_seed(42)


def attack_dataloader(dataset_name: str, sample_id, targets, transform=None):
    # Set multi-target labels
    # target0, target1 = "WARNING!", "ERROR!"
    datasets = [
        Subset(
            load_dataset(dataset_name, target=target, transform=transform), sample_id[i]
        )
        for i, target in enumerate(targets)
    ]
    dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return dataloader


def main():
    # init
    cfg = Config()
    model = get_model(cfg.model_name)
    dataloader = attack_dataloader(cfg.dataset_name, cfg.sample_id, cfg.targets)
    constraint = Constraint(cfg.attack_mode, frame_width=cfg.frame_width, ref_size=299)
    attacker = Attacker(model, constraint, cfg)
    run = WBLogger(
        project="VLM_batch_test",
        config=cfg,
        name="batch=1",
    ).run
    # accelerator = Accelerator()
    # model, dataloader = accelerator.prepare(model, dataloader)
    try:
        # train loop
        with tqdm(range(cfg.epoch)) as pbar:
            for i in pbar:
                loss = attacker.trainer(dataloader)
                # attacker.saver(f"./save/{str(i)}_0.pth")
                run.log({"loss": loss})
                pbar.set_postfix({"loss": f"{loss:.2f}"})
    finally:
        # save perturbation and mask
        attacker.saver(filename := "./save/perturbation.pth")
        run.save(filename, base_path="save")


if __name__ == "__main__":
    main()
