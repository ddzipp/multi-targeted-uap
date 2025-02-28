import importlib
import os

import timm

from models.base import RegisterModel

models_dir = os.path.dirname(__file__)
for filename in os.listdir(models_dir):
    if filename.endswith(".py") and not filename.startswith("__"):
        module_name = filename[:-3]
        module = importlib.import_module(f".{module_name}", package="models")

model_hub = RegisterModel.model_hub
model_hub = {k.lower(): v for k, v in model_hub.items()}


def get_model(model_name, *args, **kwargs):
    if model_name.lower() in model_hub:
        model = model_hub[model_name.lower()](*args, **kwargs)
        return model.model, model.processor
    else:
        model = timm.create_model(model_name, pretrained=True).eval()
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)
        return model, transform
