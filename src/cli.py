from src.models import MinimalSource, MinimalSearchResults, StudentSearchResults, MinimalAnswer, StudentSearchResultsAndAnswer, RagDataset, UnansweredQuestion, AnsweredQuestion
from src.chunking import get_chunks_and_put_in_file
from src.execptions import RAGError
from src.load import parse_questions
from pathlib import Path
import uuid
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
import bm25s


PROCESSED_DIR = Path("data/processed")


class Rag():
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

        chunks: list[MinimalSource] = get_chunks_and_put_in_file(max_chunk_size, overlap, PROCESSED_DIR)
        chunks_content: list[str] = []
        for chunk in chunks:
            chunks_content.append(chunk.chunk_content)

        chunks_content_tokens = bm25s.tokenize(chunks_content)
        bm25 = bm25s.BM25(method="bm25l", k1=1, b=0.5)
        bm25.index(chunks_content_tokens)
        bm25.save(str(PROCESSED_DIR))
        return "Indexing complete!"

    def search(self, query: str, k: int):
        """Return the top-k sources for a single query."""
        if not isinstance(query, str) :
            raise ValueError("The variable 'query' must to be"
                             "a string")
        if (not isinstance(k, int)
            or k <= 0):
            raise ValueError("The variable 'k' must to be"
                             " a positive integer")
        try:
            bm25 = bm25s.BM25.load(PROCESSED_DIR)
        except (OSError, ValueError, KeyError, TypeError):
            raise RAGError("Index BM25 not found, executes the command index!")
        chunks_path = PROCESSED_DIR / "chunks.json"
        PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
        if not chunks_path.is_file():
            raise FileNotFoundError("File 'chunks.json' not found!, executes the command index!")

        query_tokenized = bm25s.tokenize(query)
        results, scores = bm25.retrieve(query_tokenized, k=k)
        with open("data/processed/chunks.json", encoding="utf-8") as f:
            file: list[dict] = json.load(f)
        if not file:
            raise ValueError("File 'data/processed/chunks.json' can't be empty!")
        chunks: list[MinimalSource] = [MinimalSource.model_validate(chunk)
                                       for chunk in file]
        top_k_chunks: list[MinimalSource] = [chunks[i] for i, s in
                                             zip(results.tolist()[0],
                                                 scores.tolist()[0]) if s > 0]
        if not top_k_chunks:
            raise RAGError("The database does not have enough context to the query!\n")
        minimal_search_results = MinimalSearchResults(question=query,
                                                      question_id=str(uuid.uuid4()),
                                                      retrieved_sources=top_k_chunks)
        with open(PROCESSED_DIR / "search_results.json", "w", encoding="utf-8") as f:
            json.dump(minimal_search_results.model_dump(), f, indent=4)
        return minimal_search_results

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        """Run search over a whole dataset and write a StudentSearchResults JSON file."""
        if not isinstance(dataset_path, str) or not isinstance(save_directory, str):
            raise ValueError("The variable 'dataset_path' and 'save_directory"
                             "' must to be a string that represents a path")
        if (not isinstance(k, int)
            or k <= 0):
            raise ValueError("The variable 'k' must to be"
                             " a positive integer")

        list_minimal_search_results = []
        for question in parse_questions(dataset_path):
            try:
                top_k_chunks: MinimalSearchResults = self.search(question.question, k)
            except RAGError:
                print(f"The database does not have enough context to "
                        f"the question '{question.question}' "
                        f"with id '{question.question_id}' \n")
                continue
            list_minimal_search_results.append(MinimalSearchResults(question=question.question,
                                    question_id=question.question_id,
                                    retrieved_sources=top_k_chunks.retrieved_sources))
        DESTINY_DIR = Path(save_directory)
        DESTINY_DIR.mkdir(exist_ok=True, parents=True)
        file_name = "StudentSearchResultscode.json" if "code" in dataset_path else "StudentSearchResultsdocs.json"
        with open(DESTINY_DIR / file_name, "w", encoding="utf-8") as f:
            json.dump(StudentSearchResults(search_results=list_minimal_search_results, k=k).model_dump(), f, indent=4)
    
    def answer(self, query: str, k: int):
        """Answer a single query using the retrieved context."""
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        top_k_chunks_model: MinimalSearchResults = self.search(query, k)
        return self.__get_answer(query, top_k_chunks_model, tokenizer, model)

    def answer_dataset(self, student_search_results_path: str, save_directory: str):
        """Generate answers for a dataset, producing a StudentSearchResultsAndAnswerJSON file."""
        if not isinstance(student_search_results_path, str) or not isinstance(save_directory, str):
            raise ValueError("The variable 'student_search_results_path' and 'save_directory"
                             "' must to be a string that represents a path")
        # verificar se o file exitse, eu tenho uma ideia de rastrear o nome do file co erro
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        with open(student_search_results_path, encoding="utf-8") as f:
            file = json.load(f)
        search_results: StudentSearchResults = [StudentSearchResults.model_validate(file)]
        answers_list: list[MinimalAnswer] = []

        for metadata in search_results[0].search_results:
            answer = self.__get_answer(metadata.question, metadata, tokenizer, model)
            answers_list.append(MinimalAnswer(question=metadata.question,
                                              question_id=metadata.question_id,
                                              retrieved_sources=metadata.retrieved_sources,
                                              answer=answer))

        search_results_and_answer: StudentSearchResultsAndAnswer = StudentSearchResultsAndAnswer(search_results=answers_list,                                                                                 k=search_results[0].k)
        destiny_dir = Path(save_directory)
        destiny_dir.mkdir(exist_ok=True, parents=True)
        with open(destiny_dir / "StudentSearchResultsAndAnswer.json", "w", encoding="utf-8") as f:
            json.dump(search_results_and_answer.model_dump(), f, indent=4)


    def __IoU(self, eval_start_index, eval_end_index, student_start_index, student_end_index):
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
        
    def __recallok(self, eval_results, student_results):
        aproved = 0
        rejected = 0
        for evall, student in zip(eval_results, student_results):
            if any((evall.sources[0].file_path == source.file_path
               and self.__IoU(evall.sources[0].first_character_index,
                              evall.sources[0].last_character_index,
                              source.first_character_index,
                              source.last_character_index))
                              for source in student.retrieved_sources):
                aproved += 1
            else:
                rejected += 1

        return aproved, rejected

    def evaluate(self, student_search_results_path: str, dataset_path: str):
        """Report your own recall@k against a ground-truth dataset, for your own testing."""

        if not isinstance(student_search_results_path, str) or not isinstance(dataset_path, str):
            raise ValueError("The variable 'student_search_results_path' and 'dataset_path"
                            "' must to be a string that represents a path")

        with open(student_search_results_path, encoding="utf-8") as f:
            file = json.load(f)
        student_search_results_and_answer_model: StudentSearchResults = StudentSearchResults.model_validate(file)
        with open(dataset_path, encoding="utf-8") as f:
            file = json.load(f)
        rag_questions_model: RagDataset[AnsweredQuestion] = RagDataset.model_validate(file)

        eval_results = rag_questions_model.rag_questions
        student_results = student_search_results_and_answer_model.search_results
        aproved, rejected = self.__recallok(eval_results, student_results)
        total = aproved + rejected
        print("total:", total, "| aproved:", aproved, "| rejected:", rejected)
        print("Percentage:", aproved * 100 / total)


    def __get_answer(self, query: str, top_k_chunks_model: MinimalSearchResults, tokenizer, model):
        chunks_content: list[str] = [chunk.chunk_content for chunk in top_k_chunks_model.retrieved_sources]

        prompt = self.__prompt_to_qwen(tokenizer, chunks_content, query)
        prompt_tokenized = tokenizer(prompt, return_tensors="pt")
        input_len = len(prompt_tokenized["input_ids"][0]) # or prompt_tokenized["input_ids"].shape[1]

        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=250)
        answer_tokens = output_tokenized[0][input_len:]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        return answer

    @staticmethod
    def __prompt_to_qwen(tokenizer: PreTrainedTokenizer, chunks_content: list[str], query: str):
        messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a question-answering assistant for the vLLM codebase. "
                        "Answer the user's question using only the information provided "
                        "in the context. "
                        "Do not use external knowledge or invent facts, commands, "
                        "parameters, file paths, or configuration options. "
                        "If the context does not contain enough information to answer, "
                        "clearly state that the information is insufficient. "
                        "Answer directly and concisely. "
                        "Do not repeat the question or the answer. "
                        "Return only one answer, without repeating it or adding "
                        "multiple Answer sections."
                    )
                },
                {
                    "role": "user",
                    "content": (
                                f"Context:\n{'\n\n'.join(chunks_content)}"
                                f"\n\nQuestion: {query}")
                }
            ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)