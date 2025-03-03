import wandb

api = wandb.Api()
import seaborn as sns
import torch

runs = api.runs("lichangyue/multi-targeted-VLM-test")


# run = runs[0]
# files = run.files()
# list(files)
# pert = run.file("perturbation.pth").download(root="./save", replace=True)
# result = torch.load(pert.name)
# pert = result["perturbation"]
# mask = result["mask"]

histories = runs.histories(format="pandas")
sns.lineplot(data=histories, x="_step", y="loss", hue="run_id")
