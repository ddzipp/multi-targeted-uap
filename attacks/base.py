import os

import torch
from tqdm import tqdm

from dataset.base import AttackDataset
from models.base import Model, TimmModel
from utils.constraint import Constraint


class Attacker:
    def __init__(
        self,
        model: Model,
        constraint: Constraint,
        lr=0.1,
        on_normalized=True,  # add noise on normalized pixel_values
        momentum=0.9,
        bound: tuple = (0, 1),
        ref_size=299,
    ):
        super().__init__()
        self.model = model
        self.constraint = constraint
        self.base_lr = lr
        self.lr = lr * 10
        self.momentum = momentum
        self.on_normalized = on_normalized
        self.bound = bound
        self.ref_shape = (1, 3, ref_size, ref_size)
        self.pert = self.__init_pert__()
        self.velocity = torch.zeros_like(self.pert)

    def __init_pert__(self):
        self.pert = torch.rand(self.ref_shape)
        mask = self.constraint.get_mask(self.ref_shape)
        if self.on_normalized:
            self.pert = self.model.image_preprocess(self.pert)
            if self.pert.ndim == 4:
                # generate mask directly with new shape
                self.constraint.get_mask(self.pert.shape)
            else:  # let the mask go through the same image_preprocess
                self.constraint._mask = self.model.image_preprocess(mask, False)
            # init clip bound for normalized pixel_values
            lower, upper = self.bound
            min_values = lower * torch.ones(self.ref_shape)
            max_values = upper * torch.ones(self.ref_shape)
            min_values = self.model.image_preprocess(min_values)
            max_values = self.model.image_preprocess(max_values)
            self.bound = (min_values, max_values)
        return self.pert

    def clip_image(self, image: torch.Tensor):
        batch = image.shape[0] // self.pert.shape[0]
        if not self.on_normalized:
            return torch.clip(image, *self.bound)
        else:
            lower, upper = self.bound
            lower = lower.repeat((batch,) + (1,) * (image.ndim - 1))
            upper = upper.repeat((batch,) + (1,) * (image.ndim - 1))
            return torch.clip(image, lower, upper)

    def get_adv_inputs(self, inputs: dict):
        # add perturbation to normalized pixel_values
        if self.on_normalized:
            inputs["pixel_values"] = self.constraint(inputs["pixel_values"], self.pert)
        else:
            raise NotImplementedError("Only on_normalized is supported")
        inputs["pixel_values"] = self.clip_image(inputs["pixel_values"])
        return inputs

    def step(self, grad):
        self.velocity = self.momentum * self.velocity + grad / torch.norm(grad, p=1)
        self.pert = self.pert - self.lr * self.velocity.sign()
        self.pert = self.clip_image(self.pert)

    def trainer(self, dataloader) -> float:
        total_loss = 0
        total_ce_loss, total_neg_loss = 0, 0
        dataset: AttackDataset = dataloader.dataset
        target_dict = dataset.target_dict

        def calc_loss(logits, labels, targets):
            # cross entropy loss for positive targets
            loss_fn = torch.nn.CrossEntropyLoss()
            ce_loss = loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))
            # sum confidence of negative targets
            prob = logits.softmax(dim=-1)
            neg_loss = 0
            margin = 0.2
            for key, val in target_dict.items():
                neg_mask = [i != key for i in labels]
                # if all labels are positive, skip this target
                if not any(neg_mask):
                    continue
                prob_mask = prob[neg_mask].view(-1, logits.shape[-1])
                anchor_prob = prob_mask[torch.arange(prob_mask.shape[0]), targets[neg_mask].view(-1)]
                neg_prob = prob_mask[torch.arange(prob_mask.shape[0]), val * sum(neg_mask)]
                margin_loss = (neg_prob - anchor_prob + margin).clip(min=0).mean()
                neg_loss += margin_loss
            neg_loss = neg_loss / len(target_dict)
            # total loss
            loss = ce_loss + neg_loss
            # record ce_loss and neg_loss for logging
            nonlocal total_ce_loss, total_neg_loss
            total_ce_loss += ce_loss.item()
            total_neg_loss += neg_loss.item()
            return loss

        for item in dataloader:
            # self.optimizer.zero_grad()
            self.pert = self.pert.detach().requires_grad_()
            inputs = self.get_adv_inputs(item["inputs"])
            logits = self.model.calc_logits(inputs, item["targets"])
            loss = calc_loss(logits, item["labels"], item["targets"])
            # autograd can save memory by recording pert grad only
            grad = torch.autograd.grad(loss, self.pert)[0]
            self.step(grad)
            total_loss += loss.item()
        return {
            "loss": total_loss / len(dataloader),
            "ce_loss": total_ce_loss / len(dataloader),
            "neg_loss": total_neg_loss / len(dataloader),
        }

    @torch.no_grad()
    def tester(self, dataloader):
        processor, model = self.model.processor, self.model.model
        preds, targets = [], []
        for item in tqdm(dataloader, desc="Testing"):
            # label = torch.tensor([int(label) for label in item["label"]], device="cuda")
            if isinstance(self.model, TimmModel):
                logits = self.model.forward(item)
                pred = logits.argmax(-1)
            else:
                target_tokens = processor.batch_decode(item["targets"], skip_special_tokens=True)
                inputs = self.get_adv_inputs(item["inputs"])
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                output = model.generate(**inputs, max_new_tokens=5)
                pred = processor.batch_decode(output[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            # asr += (pred == targets).sum().item()
            preds += pred
            targets += target_tokens

        return preds, targets

    def saver(self, filename="./save/perturbation.pth"):
        dirpath = os.path.dirname(filename)
        os.makedirs(dirpath, exist_ok=True)
        torch.save(
            {
                "perturbation": self.pert.detach(),
                "velocity": self.velocity.detach(),
                # "mask": self.constraint.mask,
            },
            filename,
        )
