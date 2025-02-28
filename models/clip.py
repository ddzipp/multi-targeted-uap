from transformers import CLIPModel, CLIPProcessor

from models.base import RegisterModel, VisualLanguageModel


@RegisterModel()
class Clip(VisualLanguageModel):

    model_id = "openai/clip-vit-base-patch32"

    def __init__(self, device="auto", torch_dtype="float16"):
        """
        model_name=="blip2":
            loss = eval_model.model(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                pixel_values=input_x,
                attention_mask=attention_mask,
                labels=labels,
                normalize_vision_input = True
            )[0]
        """

        self.device = device
        self._model = CLIPModel.from_pretrained(
            self.model_id, device_map=device, torch_dtype=torch_dtype
        )
        self._processor = CLIPProcessor.from_pretrained(
            self.model_id, device_map=device, torch_dtype=torch_dtype
        )

    def prompt_wrap(self, question: str, answer: str | None = None) -> str:
        return f"Question:{question} Answer:{answer if answer is not None else ''}"

    def image_encode(self, pixel_values):
        # pixel values is normalized
        emb = self.model.vision_model(pixel_values)[0]
        return emb

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model
