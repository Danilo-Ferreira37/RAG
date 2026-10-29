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

        with open(student_search_results_path, encoding="utf-8") as f:
            file = json.load(f)
        student_search_results_and_answer_model: StudentSearchResultsAndAnswer = StudentSearchResultsAndAnswer.model_validate(file)
        with open(dataset_path, encoding="utf-8") as f:
            file = json.load(f)
        rag_questions_model: RagDataset[Answered] = RagDataset.model_validate(file)


        eval_results = rag_questions_model.rag_questions
        student_results = student_search_results_and_answer_model.search_results
        for evall, student in zip(eval_results, student_results):
            print(evall.question)
            print("Eval result path", evall.sources[0].file_path)
            print("\n")
            print(student.question)
            print("Student result path", student.retrieved_sources[0].file_path)
            print()
            if evall.sources[0].file_path == student.retrieved_sources[0].file_path:
                print("passed danilo\n")
            else:
                print("failed danilo\n")