from models.base import RegisterModel, VisualLanguageModel


@RegisterModel()
class Blip2(VisualLanguageModel):
    model_id = "Salesforce/blip2-opt-2.7b"

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

        # Usage
        input_ids = tokenizer(prompt, return_tensors="pt")
        pixel_values = image_processor(images, return_tensors="pt")["pixel_values"]
        output = model (.generate)(input_ids=input_ids["input_ids"],
            pixel_values=pixel_values, attention_mask=input_ids["attention_mask"])
        processor.batch_decode(output.logits[:, -1, :].argmax(-1), skip_special_tokens=True)
        """
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.device = device
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id, device_map=device, torch_dtype=torch_dtype
        )
        self._processor = Blip2Processor.from_pretrained(self.model_id)

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model

    # def prompt_wrap(self, question:str, answer:str|None=None) -> str:
    #     return f"Question:{question} Answer:{answer if answer is not None else ''}"
    #
    # def image_encode(self, pixel_values):
    #     # pixel values is normalized
    #     emb = self.model.vision_model(pixel_values)[0]
    #     return emb


@RegisterModel()
class InstructBlip2(VisualLanguageModel):
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

    model_id = "Salesforce/instructblip-vicuna-7b"

    def __init__(self, device="auto", torch_dtype="float16"):
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

        self.device = device
        self._model = InstructBlipForConditionalGeneration.from_pretrained(
            self.model_id, device_map=device, torch_dtype=torch_dtype
        )
        self._processor = InstructBlipProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer
        self.lm_embeds = self.model.get_input_embeddings()
        self.image_processor = self.processor.image_processor
        self.qformer = self.processor.qformer

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model
