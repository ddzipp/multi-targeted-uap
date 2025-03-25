from torchvision.datasets import ImageFolder

from dataset.base import VisionData


class MUAP(ImageFolder):
    def __init__(self, path, transform=None):
        super().__init__(path, transform=transform)
        self.transform = transform
        # with open(path + "/labels.json", "r") as f:
        #     self.ids_2_labels = json.load(f)["labels"]
        # self.targets = list(self.ids_2_labels.values())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index) -> VisionData:
        path, _ = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        label = str(index // 8)
        response = VisionData(
            image=sample,
            label=label,
            question="Describe this image.",
            answer="Unknown.",
        )
        return response
