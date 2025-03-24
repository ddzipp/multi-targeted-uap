import torch

from models.base import VisualLanguageModel


class OpenVLA(VisualLanguageModel):
    """
    OpenVLA model for visual language tasks.
    """

    model_id = "openvla/openvla-7b"

    def __init__(self, device="auto", torch_dtype="float16"):
        """
        Initialize the OpenVLA model.

        Args:
            device (str): Device to use for the model. Default is "auto".
            torch_dtype (str): Data type for PyTorch tensors. Default is "float16".
        """
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model
