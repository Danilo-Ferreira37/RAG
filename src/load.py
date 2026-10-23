from src.models import RagDataset
from typing import Any
from src.models import AnsweredQuestion, UnansweredQuestion, RagDataset
import json
from pydantic import ValidationError


def json_load(path: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        file = f.read()
    return json.loads(file)


def parse_questions(path: str, answered: bool = False):
    file_json: dict[str, object] = json_load(path)
    model_RagDataset: RagDataset = RagDataset.model_validate(file_json)
    model = AnsweredQuestion if answered else UnansweredQuestion
    config = []

    for q in model_RagDataset.rag_questions:
        model.model_validate(q)
        config.append(q)

    return config