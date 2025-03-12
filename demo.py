import os

import torch
from accelerate import Accelerator
from torch.utils.data import Subset
from tqdm import tqdm

from attacks import get_attacker
from config import Config
from dataset import collate_fn, load_dataset
from models import get_model
from utils.logger import WBLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.manual_seed(42)


def attack_dataloader(name: str, sample_id, targets, transform=None, split="val"):
    # Set multi-target labels
    datasets = [
        Subset(
            load_dataset(name, split=split, target=target, transform=transform),
            sample_id[i],
        )
        for i, target in enumerate(targets)
    ]
    dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return dataloader


def main():
    # init
    cfg = Config()
    model = get_model(cfg.model_name)
    dataloader = attack_dataloader(
        cfg.dataset_name, cfg.sample_id, cfg.targets, cfg.split
    )
    attacker = get_attacker(cfg, model)
    run = WBLogger(
        project="generability_test",
        config=cfg,
        name="train_20_imgs",
    ).run
    # TODO: Accelerator is not supported in this version
    # accelerator = Accelerator()
    # model, dataloader = accelerator.prepare(model, dataloader)
    # attacker.pert = accelerator.prepare(attacker.pert)
    try:
        # train loop
        with tqdm(range(cfg.epoch)) as pbar:
            for i in pbar:
                loss = attacker.trainer(dataloader)
                # attacker.saver(f"./save/{str(i)}_0.pth")
                run.log({"loss": loss})
                pbar.set_postfix({"loss": f"{loss:.2f}"})
                if loss < 0.15:
                    break
    finally:
        # save perturbation and mask
        attacker.saver(filename := "./save/perturbation.pth")
        run.save(filename, base_path="save")
        run.finish()


if __name__ == "__main__":
    main()
