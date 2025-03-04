import json
import re

import torch
import yaml

# Online
import wandb
from attacks.base import Attacker
from config import Config
from demo import attack_dataloader
from models import get_model
from utils.constraint import Constraint

# Online
api = wandb.Api()
run_path = "lichangyue/label_num_DNN_test/0ex4qqxf"
run = api.run(run_path)
config = json.loads(run.json_config)
file = run.file("perturbation.pth").download(
    root="./save", replace=False, exist_ok=True
)
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
cfg.sample_id = torch.tensor(cfg.sample_id) + 30

model = get_model(cfg.model_name)
dataloader = attack_dataloader(cfg.dataset_name, cfg.sample_id, cfg.targets)
dataset = dataloader.dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=False)

constraint = Constraint(cfg.attack_mode, frame_width=cfg.frame_width, ref_size=299)
attacker = Attacker(model, constraint, cfg)
attacker.pert = results["perturbation"]

asr = 0.0
acc = 0.0
loss = 0.0
model.model.eval()
with torch.no_grad():
    for item in dataloader:
        inputs, target = attacker.get_inputs(**item, generation=True)
        image, target = inputs["pixel_values"].cuda(), target.cuda()
        label = torch.tensor([int(label) for label in item["label"]], device="cuda")
        logits = model.model(image)
        pred = logits.argmax(-1)
        acc += (pred == label).sum().item()
        asr += (pred == target).sum().item()

print("Loss", model.calc_loss(inputs, target))
asr /= len(dataset)
acc /= len(dataset)
print("ASR: ", asr)
print("ACC: ", acc)
