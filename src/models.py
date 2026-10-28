from pydantic import BaseModel, Field, ConfigDict
from typing import List, Any
import uuid

#ja usei
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    chunk_content: str | None = None

# ja usei
class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str

# so so
class AnsweredQuestion(UnansweredQuestion):
    question_id: str
    question: str
    answer: str
    sources: List[MinimalSource]
    difficulty: str
    is_valid: bool

# ja usei
class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]







# ja usei
class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


# ja usei
class MinimalAnswer(MinimalSearchResults):
    answer: str






# ja usei
class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

# ja usei
class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
