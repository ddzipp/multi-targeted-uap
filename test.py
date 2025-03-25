import json
import re

import torch

import wandb
from attacks import get_attacker
from config import Config
from demo import get_dataloader
from models import get_model

# Online
api = wandb.Api()
run_path = "lichangyue/qwen-test/b2v4c3pt"
run = api.run(run_path)
config = json.loads(run.json_config)

f = run.file("perturbation.pth").download(root="./save", replace=True, exist_ok=True)
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
cfg.sample_id = torch.tensor([list(map(int, re.findall(r"\d+", x))) for x in cfg.sample_id[1:-2].split(r"]")])
cfg.sample_id = torch.tensor(cfg.sample_id)
model = get_model(cfg.model_name)


def evaluate(cfg: Config):
    dataloader = get_dataloader(
        cfg.dataset_name,
        cfg.sample_id,
        cfg.targets,
        split=cfg.split,
        shuffle=False,
        batch_size=1,
    )
    attacker = get_attacker(cfg, model)
    attacker.pert = results["perturbation"]

    asr = 0.0
    model.model.eval()
    asr = attacker.tester(dataloader)

    # print("Loss", loss := loss / len(dataloader))
    # acc /= len(dataloader)
    print("ASR: ", asr)
    # print("ACC: ", acc)
    return asr


asr = evaluate(cfg)
run.summary.update({"Train_ASR": asr})

cfg.sample_id = (cfg.sample_id + 28)[:12]
asr = evaluate(cfg)
run.summary.update({"Test_ASR": asr})
