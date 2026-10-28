#         with open(f"{str(PROCESSED_DIR)}/RAG_vocab.json", "w") as f:    
#             json.dump([chunks_content_tokens.vocab], f, indent=4)
#         with open(f"{str(PROCESSED_DIR)}/RAG_ids.json", "w") as f:   
#             json.dump([{f"chunk {i}": l} for i, l in enumerate(chunks_content_tokens.ids, 1)],
#                       f, indent=4)
# comparar o que é mais rapido, duas list comprehension ou apenas um loop for
#         chunks_content = [chunk.chunk_content for chunk in chunks]
#         corpus = [{
#             "file_path": chunk.file_path,
#             "first_character_index": chunk.first_character_index,
#             "last_character_index": chunk.last_character_index,
#             "chunk_content": chunk.chunk_content} for chunk in chunks]

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        """Run search over a whole dataset and write a StudentSearchResults JSON file."""
        if not isinstance(dataset_path, str) or not isinstance(save_directory, str):
            raise ValueError("The variable 'dataset_path' and 'save_directory"
                             "' must to be a string tha represents a path")
        if (not isinstance(k, int)
            or k <= 0):
            raise ValueError("The variable 'k' must to be"
                             " a positive integer")
        try:
            bm25 = bm25s.BM25.load(PROCESSED_DIR)
            chunks_path = PROCESSED_DIR / "chunks.json"
            if not chunks_path.is_file():
                raise FileNotFoundError("File 'chunks.json' not found!, executes the command index!")

            with open("data/processed/chunks.json", encoding="utf-8") as f:
                file: list[dict] = json.load(f)
            chunks = [MinimalSource.model_validate(chunk) for chunk in file]
            list_minimal_search_results = []
            for question in parse_questions(dataset_path):
                query_tokenized = bm25s.tokenize(question.question)
                results, scores = bm25.retrieve(query_tokenized, k=k)

                top_k_chunks = [chunks[i] for i, s in 
                                zip(results.tolist()[0],
                                    scores.tolist()[0]) if s > 0]
                if not top_k_chunks:
                    print(f"The database does not have enough context to "
                          f"the question '{question.question}' "
                          f"with id '{question.question_id}' \n")
                    continue
                list_minimal_search_results.append(MinimalSearchResults(question=question.question,
                                     question_id=question.question_id,
                                     retrieved_sources=top_k_chunks))
            DESTINY_DIR = Path(save_directory)
            DESTINY_DIR.mkdir(exist_ok=True, parents=True)
            with open(f"{str(DESTINY_DIR)}/StudentSearchResults.json", "w", encoding="utf-8") as f:
                json.dump(StudentSearchResults(search_results=list_minimal_search_results, k=k).model_dump(), f, indent=4)

        except FileNotFoundError:
            raise FileNotFoundError("File to dataset_path not found!")
        except (OSError, ValueError, KeyError, TypeError):
            raise RAGError("Index BM25 not found, executes the command index!")