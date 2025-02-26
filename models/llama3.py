from models.base import RegisterModel, VisualLanguageModel


@RegisterModel()
class Llama3(VisualLanguageModel):

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    def __init__(self, device="auto", torch_dtype="float16"):
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        self._model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id, device_map=device, torch_dtype=torch_dtype
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self.device = "cuda" if device == "auto" else device

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model


# class Llama3VisionModel(TransformersModel):
#     def __init__(self):
#         super().__init__()
#         model_id = "meta-llama/Llama-3.2-11B-Vision"
#         model = MllamaVisionModel.from_pretrained(model_id)
#         processor = AutoProcessor.from_pretrained(model_id)
#         url = "https://www.ilankelman.org/stopsigns/australia.jpg"
#         image = Image.open(requests.get(url, stream=True).raw)
#         inputs = processor(images=image, return_tensors="pt")
#         output = model(**inputs)
