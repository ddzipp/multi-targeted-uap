import json
import re

import torch
import yaml  # type: ignore

import wandb
from attacks import get_attacker
from config import Config
from demo import attack_dataloader
from models import get_model
from utils.constraint import Constraint

# Online
api = wandb.Api()
run_path = "lichangyue/attack_mode_test/4yl50hgb"
run = api.run(run_path)
config = json.loads(run.json_config)
file = run.file("perturbation.pth").download(root="./save", replace=True, exist_ok=True)
results = torch.load(file.name)

# Offline
# local_path = "wandb/run-20250304_061521-8og0625r"
# with open(local_path + "/files/config.yaml", "r", encoding="utf-8") as f:
#     config = yaml.safe_load(f)
# results = torch.load(local_path + "/files/perturbation.pth")


config = {key: value["value"] for key, value in config.items() if key != "_wandb"}
cfg = Config(**config)

# Test on the training set or the test set
cfg.sample_id = torch.tensor(
    [list(map(int, re.findall(r"\d+", x))) for x in cfg.sample_id[1:-2].split(r"]")]
)
cfg.sample_id = torch.tensor(cfg.sample_id)
model = get_model(cfg.model_name)


def evaluate(cfg: Config):

    dataloader = attack_dataloader(
        cfg.dataset_name, cfg.sample_id, cfg.targets, split=cfg.split
    )
    dataset = dataloader.dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=False)
    attacker = get_attacker(cfg, model)
    attacker.pert = results["perturbation"]

    asr = 0.0
    acc = 0.0
    loss = 0.0
    model.model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, item in enumerate(dataloader):
            inputs, target = attacker.get_inputs(**item, generation=True)
            image, target = inputs["pixel_values"].cuda(), target.cuda()
            label = torch.tensor([int(label) for label in item["label"]], device="cuda")
            logits = model.model(image)
            pred = logits.argmax(-1)
            acc += (pred == label).sum().item()
            asr += (pred == target).sum().item()
            loss += loss_fn(logits, target).item()

    print("Loss", loss := loss / len(dataloader))
    asr /= len(dataset)
    acc /= len(dataset)
    print("ASR: ", asr)
    print("ACC: ", acc)
    return asr, acc, loss


asr, acc, loss = evaluate(cfg)
run.summary.update({"Train_ASR": asr, "Train_ACC": acc, "Train_Loss": loss})

cfg.sample_id = (cfg.sample_id + 30)[:, :20]
asr, acc, loss = evaluate(cfg)
run.summary.update({"Test_ASR": asr, "Test_ACC": acc, "Test_Loss": loss})
