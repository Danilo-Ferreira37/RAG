from src.models import RagDataset
from typing import Any
from src.models import UnansweredQuestion, RagDataset
import json


def parse_questions(path: str) -> list[UnansweredQuestion]:
      try:
            with open(path, "r", encoding="utf-8") as f:
                  file_json: dict[str, object] = json.load(f)
            model_RagDataset: RagDataset = RagDataset.model_validate(file_json)
            return [UnansweredQuestion.model_validate(q) for q in model_RagDataset.rag_questions]
      except FileNotFoundError:
            raise FileNotFoundError("File to dataset_path not found!")
            exit(1)
