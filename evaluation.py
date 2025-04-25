import os
import re

import numpy as np
import torch

import wandb
from attacks import get_attacker
from config import Config
from demo import get_dataloader
from models import get_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Online
api = wandb.Api()
run_path = "lichangyue/ImageNet-VLM-MarginLoss/ho21tqmu"
run = api.run(run_path)
config = run.config
cfg = Config()
cfg.__dict__.update(config)

file_name = f"./save/Margin/{cfg.model_name}_T{cfg.num_targets}/perturbation.pth"
result_path = file_name.replace(".pth", "_evaluation.pth")


@torch.no_grad()
def evaluate(cfg: Config, perturbation):
    dataloader = get_dataloader(
        cfg.dataset_name,
        cfg.sample_id,
        cfg.targets,
        split=cfg.split,
        batch_size=1,
        shuffle=False,
        processor=model.processor,
        eval=True,
    )
    attacker = get_attacker(cfg, model)
    attacker.pert = perturbation

    preds, targets = attacker.tester(dataloader)
    return preds, targets


def calc_asr(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    asr_targets = []
    datasize = len(preds) // cfg.num_targets
    for i in range(cfg.num_targets):
        left = i * datasize
        right = (i + 1) * datasize
        preds_i = preds[left:right]
        targets_i = targets[left:right]
        asr_targets.append((preds_i == targets_i).mean().item())
    print("ASR for each target:", asr_targets)
    print(f"Average ASR: {np.mean(asr_targets):.4f}")
    return asr_targets


if os.path.exists(result_path):
    results = torch.load(result_path)
    train_preds, train_targets = results["train_preds"], results["train_targets"]
    test_preds, test_targets = results["test_preds"], results["test_targets"]
else:
    if not os.path.exists(file_name):
        f = run.file(file_name).download()
    results = torch.load(file_name)
    perturbation = results["perturbation"]

    # Test on the training set or the test set
    if isinstance(cfg.sample_id, str):
        cfg.sample_id = torch.tensor([list(map(int, re.findall(r"\d+", x))) for x in cfg.sample_id[1:-2].split(r"]")])
        cfg.sample_id = cfg.sample_id.flatten().tolist()

    model = get_model(cfg.model_name)
    print("Evaluation on the training set")
    train_preds, train_targets = evaluate(cfg, perturbation)
    cfg.sample_id = torch.tensor(cfg.sample_id)
    cfg.sample_id = cfg.sample_id.view(cfg.num_targets, -1)
    cfg.sample_id = (cfg.sample_id + cfg.train_size)[..., :20].flatten().tolist()
    print("Evaluation on the test set")
    test_preds, test_targets = evaluate(cfg, perturbation)
    print("Done")

    torch.save(
        {
            "train_preds": train_preds,
            "train_targets": train_targets,
            "test_preds": test_preds,
            "test_targets": test_targets,
        },
        result_path,
    )

train_asr = calc_asr(train_preds, train_targets)
contain = [train_targets[i] in train_preds[i] for i in range(len(train_targets))]
print("Contain Rate", np.mean(contain))
test_asr = calc_asr(test_preds, test_targets)
contain = [test_targets[i] in test_preds[i] for i in range(len(test_preds))]
print("Contain Rate", np.mean(contain))
if "perturbation.pth" in file_name:
    run.summary.update({"Train_ASR": train_asr, "Test_ASR": test_asr})
# run.summary.update({"Test_ASR": asr})
