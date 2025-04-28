import torch
from PIL.Image import Image
import torchvision

from models.base import RegisterModel, VisualLanguageModel


@RegisterModel()
class Qwen2(VisualLanguageModel):
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self, device="auto", torch_dtype="float16"):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, device_map=device, torch_dtype=torch_dtype
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer
        self.lm_embeds = self.model.get_input_embeddings()
        self.image_processor = self.processor.image_processor
        self.device = "cuda" if device == "auto" else device

    def image_preprocess(self, image, do_normalize=True):
        hight, width = 308, 308
        resize = torchvision.transforms.Resize((hight, width))
        image = resize(image)
        return self.processor.image_processor(image, return_tensors="pt", do_resize=False, do_rescale=False, do_normalize=do_normalize)[
            "pixel_values"
        ]


    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model


@RegisterModel()
class Qwen2_5(Qwen2):
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(self, device="auto", torch_dtype="float16"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, device_map=device, torch_dtype=torch_dtype
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer
        self.lm_embeds = self.model.get_input_embeddings()
        self.image_processor = self.processor.image_processor
        self.device = "cuda" if device == "auto" else device
