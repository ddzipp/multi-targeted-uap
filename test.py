import json
import os
import re

import torch

import wandb
from attacks import get_attacker
from config import Config
from demo import get_dataloader
from models import get_model

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

# Online
api = wandb.Api()
run_path = "lichangyue/ImageNet-VLM-Eval/rkynyh46"
run = api.run(run_path)
config = json.loads(run.json_config)

f = run.file("Llava/perturbation.pth").download(root="./save", replace=False, exist_ok=True)
results = torch.load(f.name)

# Offline
# local_path = "wandb/run-20250304_061521-8og0625r"
# with open(local_path + "/files/config.yaml", "r", encoding="utf-8") as f:
#     config = yaml.safe_load(f)
# results = torch.load(local_path + "/files/perturbation.pth")


config = {key: value["value"] for key, value in config.items() if key != "_wandb"}
cfg = Config()
cfg.__dict__.update(config)

# Test on the training set or the test set
if isinstance(cfg.sample_id, str):
    cfg.sample_id = torch.tensor([list(map(int, re.findall(r"\d+", x))) for x in cfg.sample_id[1:-2].split(r"]")])
else:
    cfg.sample_id = torch.tensor(cfg.sample_id)

model = get_model(cfg.model_name)
model.model.eval()


def evaluate(cfg: Config):
    dataloader = get_dataloader(
        cfg.dataset_name, cfg.sample_id, cfg.targets, split=cfg.split, shuffle=False, batch_size=1
    )
    attacker = get_attacker(cfg, model)
    attacker.pert = results["perturbation"]

    asr = 0.0

    preds, targets = attacker.tester(dataloader)

    # print("Loss", loss := loss / len(dataloader))
    # acc /= len(dataloader)
    print("ASR: ", asr)
    return asr


def asr(preds, targets):
    asr = 0
    for p, t in zip(preds, targets):
        if p.startswith(t):
            asr += 1


# asr = evaluate(cfg)
# run.summary.update({"Train_ASR": asr})
cfg.sample_id = cfg.sample_id.view(cfg.num_targets, -1)
cfg.sample_id = (cfg.sample_id + cfg.train_size)[..., :3].reshape(-1)
asr = evaluate(cfg)
# run.summary.update({"Test_ASR": asr})
