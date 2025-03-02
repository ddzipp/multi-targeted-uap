import torch
from torchvision import transforms

from models import model_hub


class Attacker:

    def __init__(self, model, processor=None):
        super().__init__()
        self.model = model
        self.processor = processor
        self.loss_fn = torch.nn.CrossEntropyLoss()
        if self.processor.__class__.__module__.startswith("torchvision"):
            # remove ToTensor from processor, as the image have been tensorized
            self.processor = transforms.Compose(
                [
                    t
                    for t in self.processor.transforms
                    if not isinstance(t, transforms.ToTensor)
                ]
            )
        if self.model.__class__.__module__.startswith("transformers"):
            # Set eos_token and colon_ids for VLM model
            self.eos_token = self.processor.tokenizer.eos_token
            self.colon_ids = processor.tokenizer.encode(
                "Assistant:", add_special_tokens=False
            )[-1]

    def generate_inputs(self, image, questions, answers, generation=False):
        if questions is None:
            questions = "Describe this image."

        # 为每个样本生成对话模板
        prompts = []
        for q, a in zip(questions, answers):
            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": q},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": a + self.eos_token}],
                },
            ]
            if generation:
                conv.pop(-1)
            conversation = self.processor.apply_chat_template(
                conv, add_generation_prompt=generation
            )
            prompts.append(conversation)

        inputs = self.processor(
            images=image, text=prompts, return_tensors="pt", padding=True
        )
        label_ids = inputs["input_ids"].clone()
        colons_poision = torch.where(label_ids == self.colon_ids)[-1][1::2] + 1
        label_ids[
            torch.arange(label_ids.shape[1])[None, :] <= colons_poision[:, None]
        ] = -100

        return inputs, label_ids

    def calc_loss(
        self,
        image: torch.Tensor,  # with batch
        *,
        questions: list | None = None,
        labels: list | None = None
    ):
        # calc loss for vlm model and DNN model
        if self.model.__class__.__module__.startswith("transformers"):
            # VLM model
            assert questions is not None and labels is not None
            inputs, label_ids = self.generate_inputs(image, questions, labels)
            loss = self.model(**inputs, labels=label_ids).loss
        else:
            # DNN model
            if self.processor is not None:
                processed_image = self.processor(image).cuda()
            else:
                processed_image = image.cuda()
            logits = self.model(processed_image)
            target = torch.tensor(labels).cuda()
            loss = self.loss_fn(logits, target)
        return loss
