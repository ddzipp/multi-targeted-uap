import os

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from attacks import get_attacker
from config import Config
from dataset import collate_fn, load_dataset
from models import get_model
from utils.logger import WBLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
torch.manual_seed(42)


def get_dataloader(
    name: str,
    sample_id: torch.Tensor,
    targets: dict,
    split="val",
    transform=None,
    shuffle=True,
    batch_size=5,
):
    # Set multi-target labels
    dataset = load_dataset(name, split=split, targets=targets, transform=transform)
    dataset = Subset(dataset, sample_id.flatten().tolist())
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return dataloader


def main():
    # init
    cfg = Config()
    dataloader = get_dataloader(
        cfg.dataset_name,
        cfg.sample_id,
        cfg.targets,
        split=cfg.split,
        batch_size=cfg.batch_size,
    )
    model = get_model(cfg.model_name)
    attacker = get_attacker(cfg, model)
    run = WBLogger(
        project="qwen-test",
        config=cfg,
        name="Incident_3classes",
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
                if loss < 0.3:
                    break
    finally:
        # save perturbation and mask
        attacker.saver(filename := "./save/perturbation.pth")
        run.save(filename, base_path="save")
        run.finish()


if __name__ == "__main__":
    main()
