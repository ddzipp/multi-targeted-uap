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
run_path = "lichangyue/ImageNet-VLM-Eval/hj3j2af3"
run = api.run(run_path)
config = run.config
cfg = Config()
cfg.__dict__.update(config)

f = run.file(f"{config['model_name']}/perturbation.pth").download(root="./save", replace=False, exist_ok=True)
results = torch.load(f.name)

# Test on the training set or the test set
if isinstance(cfg.sample_id, str):
    cfg.sample_id = torch.tensor([list(map(int, re.findall(r"\d+", x))) for x in cfg.sample_id[1:-2].split(r"]")])
    cfg.sample_id = cfg.sample_id.flatten().tolist()

model = get_model(cfg.model_name)
model.model.eval()


@torch.no_grad()
def evaluate(cfg: Config):
    dataloader = get_dataloader(
        cfg.dataset_name, cfg.sample_id, cfg.targets, split=cfg.split, shuffle=False, batch_size=1
    )
    attacker = get_attacker(cfg, model)
    attacker.pert = results["perturbation"]

    preds, targets = attacker.tester(dataloader)
    return preds, targets


train_preds, train_targets = evaluate(cfg)

cfg.sample_id = torch.tensor(cfg.sample_id)
cfg.sample_id = cfg.sample_id.view(cfg.num_targets, -1)
cfg.sample_id = (cfg.sample_id + cfg.train_size)[..., :20].flatten().tolist()
test_preds, test_targets = evaluate(cfg)

torch.save(
    {
        "train_preds": train_preds,
        "train_targets": train_targets,
        "test_preds": test_preds,
        "test_targets": test_targets,
    },
    f.name.replace(".pth", "_evaluation.pth"),
)

# run.summary.update({"Test_ASR": asr})
