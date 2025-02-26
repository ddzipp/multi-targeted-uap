from dataset.vqa import VQADataset
import os
import json


class CropaDataset(VQADataset):
    def __init__(
        self, path="./data/Cropa", vqa_path="./data/VQA", split="val", transform=None
    ):
        super().__init__(path=vqa_path, split=split, transform=transform)

        if split == "val":
            self.path = path
            self.question_path = os.path.join(
                self.path, f"filtered_v2_OpenEnded_mscoco_{split}2014_questions.json"
            )
            self.annotation_path = os.path.join(
                self.path, f"filtered_v2_mscoco_{split}2014_annotations.json"
            )
            self.questions = json.load(open(self.question_path, "r"))["questions"]
            self.answers = json.load(open(self.annotation_path, "r"))["annotations"]
            self.length = len(self.questions)
