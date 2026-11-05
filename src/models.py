from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """
    Represents a source chunk retrieved from a document.

    Stores the file path, character indices, and optional
    content of the chunk.
    """
    file_path: str
    first_character_index: int
    last_character_index: int
    chunk_content: str | None = None


class UnansweredQuestion(BaseModel):
    """
    Represents a question that has not been answered.

    Generates a unique identifier for each question
    when one is not explicitly provided.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    Represents a question with its expected answer and sources.

    Extends UnansweredQuestion with the reference answer,
    supporting sources, difficulty, and validity information.
    """
    question_id: str
    question: str
    answer: str
    sources: List[MinimalSource]
    difficulty: str
    is_valid: bool


class RagDataset(BaseModel):
    """
    Represents a dataset containing RAG evaluation questions.

    Stores answered and unanswered questions used for
    retrieval, answer generation, and evaluation.
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    Represents the retrieval results for a single question.

    Stores the question identifier, question text, and
    the sources retrieved by the search process.
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    Represents a question with its retrieved sources and answer.

    Extends MinimalSearchResults with the generated answer.
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    Represents the retrieval results produced by a student.

    Stores the search results for multiple questions
    and the number of sources requested per question.
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    Represents the retrieval and answer results of a student.

    Stores the generated answers, their retrieved sources,
    and the number of sources requested per question.
    """
    search_results: List[MinimalAnswer]
    k: int
