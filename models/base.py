from abc import ABC, abstractmethod

import torch
from torchvision import transforms


class RegisterModel:
    model_hub: dict = {}

    def __init__(self, name=None):
        self.name = name

    def __call__(self, cls):
        name = self.name if self.name is not None else cls.__name__
        self.model_hub[name] = cls
        return cls


class Model(ABC):
    """
    Abstract base class for models.

    Attributes:
        mean (torch.Tensor): The mean values for normalization.
        std (torch.Tensor): The standard deviation values for normalization.

    Methods:
        normalize_image(image):
            Normalizes the given image using the mean and standard deviation.

        inv_normalize_image(image):
            Inversely normalizes the given image using the mean and standard deviation.

        clip_image(image, normalized=True):
            Clips the given image to be within the valid range. If normalized is True,
            the image is clipped based on the mean and standard deviation.

        resize_image(image):
            Abstract method to resize the given image. Must be implemented by subclasses.

        calc_loss(inputs, labels):
            Abstract method to calculate the loss. Must be implemented by subclasses.
    """

    @property
    @abstractmethod
    def mean(self):
        pass

    @property
    @abstractmethod
    def std(self):
        pass

    def normalize_image(self, image):
        return transforms.Normalize(self.mean, self.mean)(image)

    def inv_normalize_image(self, image):
        inv_mean = torch.tensor([-m / s for m, s in zip(self.mean, self.std)])
        inv_std = torch.tensor([1 / s for s in self.std])
        return transforms.Normalize(inv_mean, inv_std)(image)

    @abstractmethod
    def resize_image(self, image):
        pass

    @abstractmethod
    def calc_logits(self, inputs: dict, targets: torch.Tensor):
        pass

    @abstractmethod
    def generate_inputs(self, image, questions, *, targets, generation=True):
        pass

    @abstractmethod
    def image_preprocess(self, image, do_normalize=True):
        pass

    @abstractmethod
    def forward(self, *args, **kwds):
        pass

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


class VisualLanguageModel(Model):
    @abstractmethod
    def __init__(self, device="auto", torch_dtype="float16", **kwargs):
        pass

    @property
    @abstractmethod
    def processor(self):
        pass

    @property
    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def model_id(self):
        pass

    @property
    def mean(self):
        return torch.tensor(self.processor.image_processor.image_mean)

    @property
    def std(self):
        return torch.tensor(self.processor.image_processor.image_std)

    def resize_image(self, image):
        crop_size = self.processor.image_processor.crop_size
        return transforms.Resize((crop_size["height"], crop_size["width"]))(image)

    def _calc_loss(self, logits: dict, attention_mask: torch.Tensor, labels: torch.Tensor):
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            # we use the input attention mask to shift the logits and labels, because it is 2D.
            # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
            shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
            shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device))
        return loss

    def calc_logits(self, inputs: dict, targets: torch.Tensor):
        output = self.model(**inputs)
        len_target = targets.shape[-1]
        logits = output.logits[..., -len_target - 1 : -1, :].contiguous()
        return logits

    def forward(self, inputs):
        return self.model(**inputs).logits

    def generate_inputs(self, image, questions, *, targets=None, generation=True):
        prompts = []
        for q in questions:
            conv = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": q}, {"type": "image"}],
                },
            ]
            conversation = self.processor.apply_chat_template(conv, add_generation_prompt=generation)
            prompts.append(conversation)
        # the image has already rescaled to [0, 1]
        inputs = self.processor(images=image, text=prompts, return_tensors="pt", padding=True, do_rescale=False)
        if targets is None:
            return inputs, None
        inputs["input_ids"] = torch.cat([inputs["input_ids"], targets], dim=-1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.int64)
        label_ids = inputs["input_ids"].clone()
        label_ids[:, : -targets.shape[-1]] = -100
        return inputs, label_ids

    def image_preprocess(self, image, do_normalize=True):
        return self.processor.image_processor(image, return_tensors="pt", do_rescale=False, do_normalize=do_normalize)[
            "pixel_values"
        ]


class TimmModel(Model):
    def __init__(self, model_name: str):
        import timm  # pylint: disable=import-outside-toplevel

        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=True).cuda().eval()
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)
        self.transform = transforms.Compose([t for t in transform.transforms if not isinstance(t, transforms.ToTensor)])
        assert isinstance(transform.transforms[-1], transforms.Normalize)
        self._mean, self._std = transform.transforms[-1].mean, transform.transforms[-1].std
        assert isinstance(transform.transforms[0], transforms.Resize)
        self._resize_image = transform.transforms[0]
        self.processor = None

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        return self._std

    def resize_image(self, image):
        return self._resize_image(image)

    def calc_logits(self, inputs: dict, targets: torch.Tensor):
        outputs = self.forward(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels.cuda())
        return loss

    def generate_inputs(self, image, questions, *, targets, generation=True):
        processed_image = self.transform(image)
        return {"pixel_values": processed_image}, targets

    def image_preprocess(self, image, do_normalize=True):
        transform = self.transform
        if not do_normalize:
            transform = transform.transforms[:-1]
            transform = transforms.Compose(transform)
        return transform(image)

    def forward(self, inputs):
        return self.model(inputs["pixel_values"].cuda())
