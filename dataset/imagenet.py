import json

import torchvision.datasets

from dataset.base import VisionData


class ImageNetDataset(torchvision.datasets.ImageNet):
    def __init__(self, path: str = "./data/ImageNet", split="val", transform=None):
        super().__init__(root=path, split=split, transform=transform)
        # from collections import Counter
        # import json
        # c = Counter(self.targets)

        # start_idx = torch.zeros(1000).to(int)
        # for i in range(1, 1000):
        #     start_idx[i] = c[i - 1] + start_idx[i - 1]

        # for i in range(1, 1000):
        #     a = self.targets[start_idx[i] - 1]
        #     b = self.targets[start_idx[i]]
        #     assert a != b

        # json.dump(start_idx, open("./data/ImageNet/imagenet_train_start_idx.json", "w"))

    def __getitem__(self, idx) -> VisionData:
        image, label = super().__getitem__(idx)
        response = VisionData(
            image=image,
            label=str(label),
            question="Describe this image.",
            answer="Unknown.",
        )
        return response


class ImageNetTestDataset(torchvision.datasets.ImageFolder):
    def __init__(self, path: str = "./data/ImageNet", split="test", transform=None, sort=True):
        super().__init__(root=path + "/" + split, transform=transform)
        self.image2label = json.load(open(path + "/imagenet_test_labels.json"))
        self.label2image = json.load(open(path + "/imagenet_test_label2index.json"))
        if sort:
            self.samples.sort(key=lambda x: self.image2label[x[0].split("/")[-1]])
        self.targets = list(self.image2label.values())

    def __getitem__(self, idx) -> VisionData:
        image, _ = super().__getitem__(idx)
        img_path = self.imgs[idx][0]
        label = self.image2label[img_path.split("/")[-1]]
        response = VisionData(
            image=image,
            label=label,
            question="Describe this image.",
            answer="Unknown.",
        )
        return response
