from src.chunking import get_chunks_and_put_in_file
import bm25s


class Rag():
    def __init__(self):
        self.bm25 = bm25s
        self.llm = None

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        if max_chunk_size > 2000:
            raise ValueError("The max value to 'max_chunk_size' is 2000!")
        self.max_chunk_size = max_chunk_size
        overlap = 25
        chunks = get_chunks_and_put_in_file(max_chunk_size, overlap)


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
