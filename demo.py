import os

import torch
from accelerate import Accelerator
from torch.utils.data import Subset
from tqdm import tqdm

from attacks.base import Attacker
from config.config import Config
from dataset import collate_fn, load_dataset
from dataset.base import VisionData
from models import get_model
from utils.constraint import Constraint
from utils.logger import WBLogger
from utils.optimizer import Optimizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"


def attack_dataloader(dataset_name: str):
    sample_id = torch.tensor(list(range(100, 130)) + list(range(1000, 1030)))
    dataset = load_dataset(dataset_name, target=50)
    dataset_0 = Subset(dataset, sample_id[:30])
    dataset = load_dataset(dataset_name, target=100)
    dataset_1 = Subset(dataset, sample_id[30:])
    dataset = torch.utils.data.ConcatDataset([dataset_0, dataset_1])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=60,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return dataloader


def main():
    # init dataset and dataloader
    cfg = Config()
    dataloader = attack_dataloader(cfg.dataset_name)
    model, processor = get_model(cfg.model_name)
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)

    # init logger
    run = WBLogger(project="multi-targeted-test", config=cfg, name="DNN-test").run

    # initialize attack
    attacker = Attacker(model, processor)
    perturbation = torch.rand([1, 3, 299, 299])
    constraint = Constraint(cfg.attack_mode)
    optimizer = Optimizer(
        [perturbation], method=cfg.optimizer, lr=cfg.lr, accelerator=accelerator
    )

    # define train step function
    def step(item: VisionData):
        optimizer.zero_grad()
        image, target = item["image"], item["target"]
        perturbed_image = constraint(image, perturbation)
        loss = attacker.calc_loss(perturbed_image, label=target)
        accelerator.backward(loss)
        optimizer.step()
        return loss

    # train loop
    for _ in tqdm(range(cfg.epoch)):
        total_loss = 0
        for item in dataloader:
            loss = step(item)
            total_loss += loss.item()
        run.log({"loss": total_loss / len(dataloader)})
    # save perturbation and mask
    torch.save(
        {
            "perturbation": perturbation,
            "mask": constraint.mask,
        },
        "save/perturbation.pth",
    )
    run.save("save/perturbation.pth", base_path="save")


if __name__ == "__main__":
    main()
