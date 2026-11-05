from src.models import AnsweredQuestion, MinimalSearchResults
from typing import Sequence


def IoU(
        eval_start_index: int,
        eval_end_index: int,
        student_start_index: int,
        student_end_index: int
) -> bool:
    """
    Calculate the Intersection over Union (IoU) between two chunks.
    Compares the character ranges of an evaluation source and a
    student source to measure their overlap.
    Returns True if the IoU is at least 0.05, otherwise False.
    """
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
    """
    Evaluate retrieval results against the reference dataset.
    Compares the first expected source of each question with
    the student's retrieved sources using file paths and IoU.
    Returns the number of approved and rejected questions.
    """
    aproved = 0
    rejected = 0

    for evall, student in zip(eval_results, student_results):
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
