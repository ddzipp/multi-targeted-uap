import atexit

import wandb


class WBLogger:
    """Singleton class for logging to Weights and Biases.

    Returns:
        _type_: WBLogger: A singleton instance of WBLogger.
    """

    _instance = None
    _run = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._run = wandb.init(*args, **kwargs)
            atexit.register(cls._safe_finish)
        return cls._instance

    @classmethod
    def _safe_finish(cls):
        if cls._run and wandb.run:
            cls._run.finish()
            cls._run = None

    @property
    def run(self):
        return self._run

    def __del__(self):
        self._safe_finish()


class NullLogger:
    """
    A class that does nothing when called or when an attribute is accessed.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return self.__class__()
