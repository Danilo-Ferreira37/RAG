from tqdm import tqdm
from src.models import AnsweredQuestion, MinimalSearchResults
from typing import Sequence


def IoU(
        eval_start_index: int,
        eval_end_index: int,
        student_start_index: int,
        student_end_index: int
) -> bool:
    eval_range = eval_end_index - eval_start_index
    student_range = student_end_index - student_start_index
    inter_start = max(eval_start_index, student_start_index)
    inter_end = min(eval_end_index, student_end_index)
    inter = max(0, inter_end - inter_start)
    union = eval_range + student_range - inter
    iou = inter / union if union > 0 else 0
    if iou >= 0.05:
        return True
    return False


def recallok(
        eval_results: Sequence[AnsweredQuestion],
        student_results: Sequence[MinimalSearchResults]
) -> tuple[int, int]:
    aproved = 0
    rejected = 0

    for evall, student in tqdm(zip(eval_results, student_results),
                               desc="Evaluating the student's retrieve"):
        if any((evall.sources[0].file_path == source.file_path
                and IoU(evall.sources[0].first_character_index,
                        evall.sources[0].last_character_index,
                        source.first_character_index,
                        source.last_character_index))
                for source in student.retrieved_sources):
            aproved += 1
        else:
            rejected += 1

    return aproved, rejected
