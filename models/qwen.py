import torch
from PIL.Image import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

from models.base import RegisterModel, VisualLanguageModel


@RegisterModel()
class Qwen(VisualLanguageModel):
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self, device="auto", torch_dtype="float16"):

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, device_map=device, torch_dtype=torch_dtype
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer
        self.lm_embeds = self.model.get_input_embeddings()
        self.image_processor = self.processor.image_processor
        self.device = "cuda" if device == "auto" else device

    def image_encode(self, image: Image | torch.Tensor):
        processed_image = self.processor.image_processor(image, return_tensors="pt").to(
            self.device
        )
        emb = self.model.vision_model(
            processed_image["pixel_values"],
            processed_image["aspect_ratio_ids"],
            processed_image["aspect_ratio_mask"],
        )
        return emb.last_hidden_state

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model


class Qwen2__5(Qwen):
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(self, device="auto", torch_dtype="float16"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer
        self.lm_embeds = self.model.get_input_embeddings()
        self.image_processor = self.processor.image_processor
        self.device = "cuda" if device == "auto" else device
