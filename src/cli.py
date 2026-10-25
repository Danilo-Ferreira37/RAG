from src.chunking import get_chunks_and_put_in_file
import bm25s
import json
import time


class Rag():
    def __init__(self):
        self.llm = None
        self.bm25 = bm25s.BM25

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        if (not isinstance(max_chunk_size, int)
            or max_chunk_size > 2000
            or max_chunk_size <= 0):
            raise ValueError("The variable 'max_chunk_size' must to be"
                             " a positive integer and the max value "
                             "is 2000!")
        self.max_chunk_size = max_chunk_size
        overlap = 25

        chunks, PROCESSED_DIR = get_chunks_and_put_in_file(max_chunk_size, overlap)
        chunks_content: list[str] = []
        corpus: list[dict[str, int | str]] = []
        for chunk in chunks:
            chunks_content.append(chunk.chunk_content)
            corpus.append({
                "file_path": chunk.file_path,
                "first_character_index": chunk.first_character_index,
                "last_character_index": chunk.last_character_index,
                "chunk_content": chunk.chunk_content})

        chunks_content_tokens = bm25s.tokenize(chunks_content)
        self.bm25 = bm25s.BM25(corpus=corpus)
        self.bm25.index(chunks_content_tokens)
        self.bm25.save(str(PROCESSED_DIR))


    def search(self, k: int):
        """Return the top-k sources for a single query."""
        self.k = k
        return k

    def search_datase(self, dataset_path: str, k: int, save_directory: str) -> None:
        """Run search over a whole dataset and write a StudentSearchResults JSON file."""
        self.dataset_path = dataset_path
        self.k = k
        self.save_directory = save_directory
        return dataset_path, k, save_directory

    def answer(self, k: int):
        """Answer a single query using the retrieved context."""
        self.k = k
        return k

    def answer_dataset(self, student_search_results_path: str, save_directory: str):
        """Generate answers for a dataset, producing a StudentSearchResultsAndAnswerJSON file."""
        self.student_search_results_path = student_search_results_path
        self.save_directory = save_directory
        return student_search_results_path, save_directory

    def evaluate(self, student_search_results_path: str, dataset_path: str):
        """Report your own recall@k against a ground-truth dataset, for your own testing."""
        self.student_search_results_path = student_search_results_path
        self.dataset_path = dataset_path
        return student_search_results_path, dataset_path
