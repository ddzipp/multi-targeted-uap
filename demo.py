import json
import os
import random

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import wandb
from attacks import get_attacker
from config import Config
from dataset import AttackDataset, collate_fn, load_dataset
from models import get_model, model_hub
from utils.logger import WBLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.manual_seed(42)


def get_dataloader(
    name: str,
    sample_id: list,
    targets: dict[str],
    split="val",
    shuffle=True,
    batch_size=5,
    transform=None,
    processor=None,
    eval=False,
):
    # Set multi-target labels
    dataset = load_dataset(name, split=split, targets=targets, transform=transform)
    dataset = Subset(dataset, sample_id)
    dataset = AttackDataset(dataset, targets, processor, eval=eval)
    dataloader = DataLoader(dataset, batch_size, shuffle, collate_fn=collate_fn)
    return dataloader


def set_target_sample(cfg: Config, resume_id=None):
    if resume_id is not None:
        api = wandb.Api()
        run = api.run(resume_id)
        config = run.config
        cfg.__dict__.update(config)
        return cfg

    if cfg.dataset_name == "ImageNet":
        rand_targets = random.sample(range(0, 1000), cfg.num_targets * 2)
        cfg.targets = {str(i): j for i, j in zip(rand_targets[::2], rand_targets[1::2])}
        with open("./data/ImageNet/imagenet_train_start_idx.json", "r") as f:
            start_idx = json.load(f)
        sample_id = []
        for key, value in cfg.targets.items():
            sample_id += list(range(start_idx[key], start_idx[key] + cfg.train_size))
        cfg.sample_id = sample_id
        # Set targets for VLM
        if cfg.model_name.lower() in model_hub:
            with open("./data/ImageNet/idx2class.json", "r") as f:
                idx2class = json.load(f)
            for key, value in cfg.targets.items():
                cfg.targets[key] = idx2class[str(value)].split(",")[0].strip()
        return cfg


def main():
    # init
    cfg = Config()
    set_target_sample(cfg, "lichangyue/ImageNet-VLM-MarginLoss/rofx0ve0")
    run = WBLogger(project="ImageNet-VLM-MarginLoss", config=cfg, name=f"{cfg.model_name}_T{cfg.num_targets}-10*LR").run
    # cfg.epoch = 10
    model = get_model(cfg.model_name)
    dataloader = get_dataloader(
        cfg.dataset_name,
        cfg.sample_id,
        cfg.targets,
        split=cfg.split,
        batch_size=cfg.batch_size,
        processor=model.processor,
    )
    cfg.save_dir = f"./save/Margin/{cfg.model_name}_T{cfg.num_targets}"
    attacker = get_attacker(cfg, model)
    # attacker.pert = torch.load(f"./save/{cfg.model_name}_T{cfg.num_targets}/perturbation.pth")["perturbation"]
    # TODO: Accelerator is not supported in current version
    # accelerator = Accelerator()
    # model, dataloader = accelerator.prepare(model, dataloader)
    # attacker.pert = accelerator.prepare(attacker.pert)
    try:
        # train loop
        with tqdm(range(1, 1 + cfg.epoch)) as pbar:
            for i in pbar:
                loss = attacker.trainer(dataloader)
                run.log(loss)
                pbar.set_postfix({"loss": f"{loss['loss']:.2f}"})
                attacker.saver(filename := f"{cfg.save_dir}/{str(i)}.pth")
                attacker.lr = max(attacker.lr * 0.9, attacker.base_lr)
                if i % 10 == 0:
                    run.save(filename)
                # if loss < 0.1:
                #     break
    finally:
        # save perturbation and mask
        attacker.saver(filename := f"{cfg.save_dir}/perturbation.pth")
        run.save(filename)
        run.finish()


if __name__ == "__main__":
    main()
