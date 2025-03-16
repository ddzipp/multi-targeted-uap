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

    def clip_image(self, image: torch.Tensor, normalized=True, bound=(0, 1)):
        if normalized:
            min_values = ((torch.ones(3) * bound[0] - self.mean) / self.std).max()
            max_values = ((torch.ones(3) * bound[1] - self.mean) / self.std).min()
            return image.clip(min_values.to(image.device), max_values.to(image.device))
        return image.clip(bound[0], bound[1])

    @abstractmethod
    def resize_image(self, image):
        pass

    @abstractmethod
    def calc_loss(self, inputs: dict, labels: torch.Tensor):
        pass

    @abstractmethod
    def generate_inputs(self, image, *, questions, targets, generation=False):
        pass

    @abstractmethod
    def image_preprocess(self, image):
        pass


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

    def calc_loss(self, inputs: dict, labels: torch.Tensor):
        loss = self.model(**inputs, labels=labels).loss
        return loss

    def generate_inputs(self, image, *, questions, targets, generation=False):
        eos_token = self.processor.tokenizer.eos_token
        colon_ids = self.processor.tokenizer.encode(
            "Assistant:", add_special_tokens=False
        )[-1]

        prompts = []
        for q, a in zip(questions, targets):
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
                    "content": [{"type": "text", "text": a + eos_token}],
                },
            ]
            if generation:
                conv.pop(-1)
            conversation = self.processor.apply_chat_template(
                conv, add_generation_prompt=generation
            )
            prompts.append(conversation)

        inputs = self.processor(
            images=image,
            text=prompts,
            return_tensors="pt",
            padding=True,
            do_rescale=False,  # the image is already rescaled to [0, 1]
        )
        label_ids = inputs["input_ids"].clone()
        colons_poision = torch.where(label_ids == colon_ids)[-1][1::2] + 1
        # label_ids[
        #     torch.arange(label_ids.shape[1])[None, :] <= colons_poision[:, None]
        # ] = -100
        label_ids[:, :-10] = -100

        return inputs, label_ids

    def image_preprocess(self, image):
        return self.processor.image_processor(
            image, return_tensors="pt", do_rescale=False, do_normalize=False
        )["pixel_values"]


class TimmModel(Model):

    def __init__(self, model_name: str):
        import timm  # pylint: disable=import-outside-toplevel

        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=True).cuda().eval()
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)
        self.transform = transforms.Compose(
            [t for t in transform.transforms if not isinstance(t, transforms.ToTensor)]
        )
        assert isinstance(transform.transforms[-1], transforms.Normalize)
        self._mean, self._std = (
            transform.transforms[-1].mean,
            transform.transforms[-1].std,
        )
        assert isinstance(transform.transforms[0], transforms.Resize)
        self._resize_image = transform.transforms[0]

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        return self._std

    def resize_image(self, image):
        return self._resize_image(image)

    def calc_loss(self, inputs: dict, labels: torch.Tensor):
        image, labels = inputs["pixel_values"].cuda(), labels.cuda()
        outputs = self.model(image)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        return loss

    def generate_inputs(self, image, *, questions, targets, generation=False):
        processed_image = self.transform(image)
        return {"pixel_values": processed_image}, torch.tensor(targets)

    def image_preprocess(self, image):
        transform = transforms.Compose(self.transform.transforms[:-1])
        return transform(image)
