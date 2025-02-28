import torch
from torchvision import transforms

from models import model_hub


class Attacker:

    def __init__(self, model, processor):
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
        else:
            # Set eos_token and colon_ids for VLM model
            self.eos_token = self.processor.tokenizer.eos_token
            self.colon_ids = processor.tokenizer.encode(
                "Assistant:", add_special_tokens=False
            )[-1]

    def generate_inputs(self, image, question, answer, generation=False):
        if question is None:
            question = "Describe this image."

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer + self.eos_token}],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False
        )

        if generation:
            conversation.pop(-1)
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        label_ids = inputs["input_ids"].clone()
        label_ids[0, : torch.where(label_ids == self.colon_ids)[-1][-1] + 1] = -100

        return inputs, label_ids

    def calc_loss(self, image, question=None, *, answer=None, label=None):
        # calc loss for vlm model and DNN model
        if self.model.__class__.__name__ in model_hub:
            # VLM model
            assert question is not None, "Question must be provided for VLM model"
            inputs, label_ids = self.generate_inputs(image, question, answer)
            loss = self.model(**inputs, labels=label_ids).loss
        else:
            # DNN model
            processed_image = self.processor(image).unsqueeze(0).cuda()
            logits = self.model(processed_image)
            target = torch.tensor([int(label)]).cuda()
            loss = self.loss_fn(logits, target)
        return loss
