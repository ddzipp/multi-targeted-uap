import os

import torch
import torchvision
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

norm_fn = torchvision.transforms.Normalize(
    mean=torch.tensor([0.4850, 0.4560, 0.4060]),
    std=torch.tensor([0.2290, 0.2240, 0.2250]),
)


def attack_dataloader(dataset_name: str, transform=None):
    sample_id = torch.tensor(list(range(0, 30)) + list(range(1000, 1030)))
    # One target is 466, another is 486
    dataset = load_dataset(dataset_name, target=466, transform=transform)
    dataset_0 = Subset(dataset, sample_id[:30])
    dataset = load_dataset(dataset_name, target=486, transform=transform)
    dataset_1 = Subset(dataset, sample_id[30:])
    dataset = torch.utils.data.ConcatDataset([dataset_0, dataset_1])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=60,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return dataset, dataloader


def main():
    # init
    cfg = Config()
    model, processor = get_model(cfg.model_name)
    dataset, dataloader = attack_dataloader(cfg.dataset_name, transform=processor)
    attacker = Attacker(model)
    constraint = Constraint(cfg.attack_mode, frame_width=cfg.frame_width)
    run = WBLogger(
        project="multi-targeted-test", config=cfg, name="double-targeted-frame30"
    ).run

    # accelerator for multi-gpu training
    # accelerator = Accelerator()
    # model, dataloader = accelerator.prepare(model, dataloader)

    momentum = 0
    torch.manual_seed(42)
    perturbation = torch.rand([1, 3, 224, 224])
    # optimizer = Optimizer([perturbation], method="adam", lr=1.0)

    # define train step function
    def step(item: VisionData, perturbation):
        image, target = item["image"], item["target"]
        perturbed_image = constraint(image, perturbation)
        loss = attacker.calc_loss(perturbed_image, label=target)
        return loss

    # train loop
    with tqdm(range(cfg.epoch)) as pbar:
        for _ in pbar:
            total_loss = 0
            for item in dataloader:
                # optimizer.zero_grad()
                perturbation = perturbation.detach().requires_grad_()
                adv_pt = norm_fn(perturbation)
                loss = step(item, adv_pt)
                loss.backward()
                # optimizer.step()
                grad = perturbation.grad
                momentum = 0.9 * momentum + grad / torch.norm(grad, p=1)
                perturbation = (perturbation - cfg.lr * momentum.sign()).clip_(0, 1)
                total_loss += loss.item()
            run.log({"loss": total_loss / len(dataloader)})
            pbar.set_postfix({"loss": f"{total_loss / len(dataloader):.2f}"})

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
