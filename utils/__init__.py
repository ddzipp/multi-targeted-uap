import torch
from torchvision.transforms import Normalize

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

## Clip perturbation
# mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
# std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

inv_mean = torch.tensor([-m / s for m, s in zip(mean, std)])
inv_std = torch.tensor([1 / s for s in std])

# The bound for the pixel values is [0, 1] in the normalized space
min_values = ((torch.zeros(3) - mean) / std).view(3, 1, 1)
max_values = ((torch.ones(3) - mean) / std).view(3, 1, 1)


def clip_image(normalized_image, normalized=False):
    if not normalized:
        return torch.clamp(normalized_image, 0, 1)
    return torch.clamp(normalized_image, min_values, max_values)


norm_fn = Normalize(mean, std)
inv_norm_fn = Normalize(inv_mean, inv_std)
