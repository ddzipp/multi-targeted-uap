import torch


class Attacker:

    def __init__(
        self,
        processor,
    ):
        super().__init__()
        self.processor = processor

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
