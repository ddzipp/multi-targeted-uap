import json
import os

from PIL import Image
from torch.utils.data import Dataset

from dataset.base import VisionData


class VQADataset(Dataset):
    def __init__(self, transform=None, path: str = "./data/VQA", split="val"):
        super().__init__()
        self.path = path
        self.split = split
        self.question_path = os.path.join(self.path, f"v2_OpenEnded_mscoco_{split}2014_questions.json")
        self.annotation_path = os.path.join(self.path, f"v2_mscoco_{split}2014_annotations.json")
        self.complicated_path = os.path.join(self.path, f"v2_mscoco_{split}2014_complicated.json")
        self.image_path = os.path.join(self.path, f"{split}2014")
        self.questions, self.answers = self.read_question_answer()
        self.length = len(self.questions)
        self.transform = transform

    def read_question_answer(self):
        with open(self.question_path, "r", encoding="utf-8") as q_file:
            questions = json.load(q_file)["questions"]
        with open(self.annotation_path, "r", encoding="utf-8") as a_file:
            answers = json.load(a_file)["annotations"]
        return questions, answers

    def get_img_path(self, question):
        return os.path.join(self.image_path, f"COCO_{self.split}2014_{question['image_id']:012d}.jpg")

    def __getitem__(self, idx) -> VisionData:
        question = self.questions[idx]
        answers = self.answers[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        if self.transform:
            image = self.transform(image)

        response = VisionData(
            image=image,
            question=question["question"],
            answer=answers["answers"][0]["answer"],
            label=None,
        )
        return response

    def __len__(self):
        return self.length
