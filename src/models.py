from pydantic import BaseModel, Field, ConfigDict
from typing import List, Any


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    chunk_content: str | None = None


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    question_id: str
    question: str
    answer: str
    sources: List[MinimalSource]
    difficulty: str
    is_valid: bool


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
    model_config = ConfigDict(extra="forbid")








class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    answer: str







class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
