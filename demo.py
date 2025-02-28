import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from attacks.base import Attacker
from config.config import Config
from dataset.base import load_dataset
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
    # init logger
    # run = WBLogger(project="multi-targeted-test", config=cfg).run
    # initialize attack
    attacker = Attacker(model, processor)

    # init perturbation, constraint and optimizer
    perturbation = torch.rand_like(dataset[0]["image"])
    constraint = Constraint(cfg.attack_mode)
    optimizer = Optimizer([perturbation], cfg.optimizer, cfg.lr)

    # main loop
    pbar = tqdm(range(cfg.epoch))

    for _ in pbar:
        item = dataset[0]
        image, question, answer = item["image"], item["question"], item["answer"]
        perturbed_image = constraint(image, perturbation)
        optimizer.zero_grad()
        loss = attacker.calc_loss(perturbed_image, label=1)
        loss.backward()
        optimizer.step()
        print(loss)


if __name__ == "__main__":
    main()
