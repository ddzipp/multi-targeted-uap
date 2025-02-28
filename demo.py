import os

import torch
from torch.utils.data import Subset
from tqdm import tqdm

from attacks.base import Attacker
from config.config import Config
from dataset import load_dataset
from dataset.base import VisionData
from models import get_model
from utils.constraint import Constraint
from utils.logger import WBLogger
from utils.optimizer import Optimizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"


def main():
    # init config, model and processor
    cfg = Config()
    model, processor = get_model(cfg.model_name)
    # init dataset and dataloader
    dataset = load_dataset(cfg.dataset_name)
    # random choose two targets and classes
    sample_id = torch.tensor(list(range(100, 130)) + list(range(1000, 1030)))
    dataset_label = Subset(dataset, sample_id)
    target0, target1 = 50, 651
    # init logger
    run = WBLogger(project="multi-targeted-test", config=cfg, name="DNN-test").run
    # initialize attack
    attacker = Attacker(model, processor)

    # init perturbation, constraint and optimizer
    perturbation = torch.rand_like(dataset[0]["image"])
    constraint = Constraint(cfg.attack_mode)
    optimizer = Optimizer([perturbation], cfg.optimizer, cfg.lr)

    # define train step function
    def step(item: VisionData, target: str | int):
        optimizer.zero_grad()
        image, question, answer = item["image"], item["question"], item["answer"]
        perturbed_image = constraint(image, perturbation)
        loss = attacker.calc_loss(perturbed_image, label=target)
        loss.backward()
        optimizer.step()
        run.log({"loss": loss.item()})
        return loss.item()

    try:
        # train loop
        with tqdm(range(cfg.epoch)) as pbar:
            for t in pbar:
                train_id = torch.randperm(len(dataset_label))
                for i in train_id:
                    item = dataset_label[i]
                    target = target0 if i < len(sample_id) // 2 else target1
                    loss = step(item, target)
                pbar.set_postfix({"loss": f"{loss:.2f}"})
                pbar.update()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # save perturbation and mask
        torch.save(
            {
                "perturbation": perturbation,
                "mask": constraint.mask,
                "sample_ids": sample_id,
                "target_0": target0,
                "target_1": target1,
            },
            "save/perturbation.pth",
        )
        run.save("save/perturbation.pth", base_path="save")


if __name__ == "__main__":
    main()
