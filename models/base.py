from abc import ABC, abstractmethod


class RegisterModel:
    model_hub: dict = {}

    def __init__(self, name=None):
        self.name = name

    def __call__(self, cls):
        name = self.name if self.name is not None else cls.__name__
        self.model_hub[name] = cls
        return cls


class Model(ABC):
    pass


class VisualLanguageModel(Model):

    @abstractmethod
    def __init__(self, device="auto", torch_dtype="float16", **kwargs):
        pass

    @property
    @abstractmethod
    def processor(self):
        pass

    @property
    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def model_id(self):
        pass
