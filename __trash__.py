        with open(f"{str(PROCESSED_DIR)}/RAG_vocab.json", "w") as f:    
            json.dump([chunks_content_tokens.vocab], f, indent=4)
        with open(f"{str(PROCESSED_DIR)}/RAG_ids.json", "w") as f:   
            json.dump([{f"chunk {i}": l} for i, l in enumerate(chunks_content_tokens.ids, 1)],
                      f, indent=4)
comparar o que é mais rapido, duas list comprehension ou apenas um loop for
        chunks_content = [chunk.chunk_content for chunk in chunks]
        corpus = [{
            "file_path": chunk.file_path,
            "first_character_index": chunk.first_character_index,
            "last_character_index": chunk.last_character_index,
            "chunk_content": chunk.chunk_content} for chunk in chunks]