import importlib
import os
import time

from models.base import RegisterModel, TimmModel

models_dir = os.path.dirname(__file__)
for filename in os.listdir(models_dir):
    if filename.endswith(".py") and not filename.startswith("__"):
        module_name = filename[:-3]
        module = importlib.import_module(f".{module_name}", package="models")

model_hub = RegisterModel.model_hub
model_hub = {k.lower(): v for k, v in model_hub.items()}


def get_model(model_name, *args, **kwargs):
    for _ in range(3):
        try:
            if model_name.lower() in model_hub:
                model = model_hub[model_name.lower()](*args, **kwargs)
                return model

            model = TimmModel(model_name)
            return model
        except Exception as e:
            last_exception = e
            time.sleep(1)
    raise last_exception
