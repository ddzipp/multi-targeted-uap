import torchvision.datasets

from dataset.base import VisualDict


class ImageNetDataset(torchvision.datasets.ImageNet):
    def __init__(self, path: str = "./data/ImageNet", split="val", transform=None):
        super().__init__(root=path, split=split, transform=transform)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        response = VisualDict(image=image, label=str(label), question=None, answer=None)
        return response
