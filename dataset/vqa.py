from torch.utils.data import Dataset
import os
import json
from PIL import Image

from dataset.base import VisualDict


class VQADataset(Dataset):
    def __init__(self, transform=None, path: str = "./data/VQA", split="val"):
        super().__init__()
        self.path = path
        self.split = split
        self.question_path = os.path.join(
            self.path, f"v2_OpenEnded_mscoco_{split}2014_questions.json"
        )
        self.annotation_path = os.path.join(
            self.path, f"v2_mscoco_{split}2014_annotations.json"
        )
        self.complicated_path = os.path.join(
            self.path, f"v2_mscoco_{split}2014_complicated.json"
        )
        self.image_path = os.path.join(self.path, f"{split}2014")
        self.questions = json.load(open(self.question_path, "r"))["questions"]
        self.answers = json.load(open(self.annotation_path, "r"))["annotations"]
        self.length = len(self.questions)
        self.transform = transform

    def get_img_path(self, question):
        return os.path.join(
            self.image_path, f"COCO_{self.split}2014_{question['image_id']:012d}.jpg"
        )

    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        if self.transform:
            image = self.transform(image)

        response = VisualDict(
            image=image,
            question=question["question"],
            answer=answers["answers"][0]["answer"],
            label=None,
        )
        return response

    def __len__(self):
        return self.length
