import os

import torchvision

from attacks.base import Attacker

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.logger import WBLogger
from utils.optimizer import Optimizer
from utils.constraint import Constraint
from config.config import Config
from models import get_model
from dataset.base import load_dataset


def main():
    # init config, model and processor
    cfg = Config()
    model, processor = get_model(cfg.model_name)
    # init dataset and dataloader
    dataset = load_dataset(cfg.dataset_name)
    # init logger
    # run = WBLogger(project="multi-targeted-test", config=cfg).run
    # initialize attack
    attacker = Attacker(processor)

    # init perturbation, constraint and optimizer
    perturbation = torch.rand_like(dataset[0]["image"])
    constraint = Constraint(cfg.attack_mode)
    optimizer = Optimizer([perturbation], cfg.optimizer, cfg.lr)

    # main loop
    pbar = tqdm(range(cfg.epoch))

    for _ in pbar:
        item = dataset[0]
        optimizer.zero_grad()
        image, question, answer = item["image"], item["question"], item["answer"]
        perturbed_image = constraint(image, perturbation)
        inputs, label_ids = attacker.generate_inputs(
            perturbed_image, question, "TARGET!"
        )
        loss = model(**inputs, labels=label_ids).loss
        loss.backward()
        optimizer.step()
        print(loss)


if __name__ == "__main__":
    main()
