import torchvision
from models.base import RegisterModel, VisualLanguageModel


@RegisterModel()
class LLava(VisualLanguageModel):
    model_id = "llava-hf/llava-1.5-7b-hf"

    def __init__(self, device="auto", torch_dtype="float16"):
        """Usage
        input_ids = tokenizer(prompt, return_tensors="pt")
        pixel_values = image_processor(images, return_tensors="pt")["pixel_values"]
        output = model (.generate)(input_ids=input_ids["input_ids"],
        pixel_values=pixel_values, attention_mask=input_ids["attention_mask"])
        processor.batch_decode(output.logits[:, -1, :].argmax(-1), skip_special_tokens=True)
        """
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.device = device
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, device_map=device, torch_dtype=torch_dtype
        )

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model


@RegisterModel()
class LLavaNext(VisualLanguageModel):
    """
    loss = eval_model.model(
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        pixel_values=input_x,
        attention_mask=attention_mask,
        labels=labels,
        normalize_vision_input = True,
        qformer_input_ids = qformer_input_ids_list[text_idx],
        qformer_attention_mask= qformer_attention_mask_list[text_idx]
    )[0]

    """

    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    def __init__(self, device="auto", torch_dtype="float16"):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        self.device = device
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, device_map=device, torch_dtype=torch_dtype
        )
        self._processor = LlavaNextProcessor.from_pretrained(self.model_id)

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model

    def image_preprocess(self, image, do_normalize=True):
        hight, width = 336,336
        resize = torchvision.transforms.Resize((hight, width))
        image = resize(image)
        return self.processor.image_processor(image, return_tensors="pt", do_resize=False, do_rescale=False, do_normalize=do_normalize)[
            "pixel_values"
        ]