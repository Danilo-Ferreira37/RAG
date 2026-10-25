from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import PythonCodeTextSplitter
from langchain_text_splitters import MarkdownTextSplitter
from pathlib import Path
from src.models import MinimalSource
import os
import json


DATA_DIR = Path("data/raw/vllm-0.10.1")
PROCESSED_DIR = Path("data/processed")


def get_chunking_of_python_files(max_chunk_size: int, overlap: int, files_content: list[str], files_path: list[str]):
    splitter = PythonCodeTextSplitter(chunk_size=max_chunk_size,
                                      chunk_overlap=overlap,
                                      add_start_index=True
                                      )
    chunks = splitter.create_documents(files_content, metadatas=files_path)
    for chunk in chunks:
        chunk.metadata["end_index"] = len(chunk.page_content) + chunk.metadata.get("start_index") - 1
    return chunks


def get_chunking_of_txt_files(max_chunk_size: int, overlap: int, files_content: list[str], files_path: list[str]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size,
                                              chunk_overlap=overlap,
                                              add_start_index=True,
                                              )
    chunks = splitter.create_documents(files_content, metadatas=files_path)
    for chunk in chunks:
        chunk.metadata["end_index"] = len(chunk.page_content) + chunk.metadata.get("start_index") - 1 
    return chunks


def get_chunking_of_md_files(max_chunk_size: int, overlap: int, files_content: list[str], files_path: list[str]):
    splitter = MarkdownTextSplitter(chunk_size=max_chunk_size,
                                    chunk_overlap=overlap,
                                    add_start_index=True,
                                    )
    chunks = splitter.create_documents(files_content, metadatas=files_path)
    for chunk in chunks:
        chunk.metadata["end_index"] = len(chunk.page_content) + chunk.metadata.get("start_index") - 1 
    return chunks


def get_chunks_and_put_in_file(max_chunk_size: int, overlap: int):
    chunks_py = []
    chunks_md = []
    chunks_txt = []
    file_path_py = []
    file_path_md = []
    file_path_txt = []

    for file_path in DATA_DIR.rglob("*"):
        if file_path.is_file and file_path.suffix == ".py":
            with open(file_path) as f:
                chunks_py.append(f.read())
                file_path_py.append({"file_path": file_path._str})

        elif file_path.is_file and file_path.suffix == ".md":
            with open(file_path) as f:
                chunks_md.append(f.read())
                file_path_md.append({"file_path": file_path._str})

        elif file_path.is_file and file_path.suffix == ".txt":
            with open(file_path) as f:
                chunks_txt.append(f.read())
                file_path_txt.append({"file_path": file_path._str})

    chunks = [*get_chunking_of_python_files(max_chunk_size, overlap, chunks_py, file_path_py),
              *get_chunking_of_txt_files(max_chunk_size, overlap, chunks_txt, file_path_txt),
              *get_chunking_of_md_files(max_chunk_size, overlap, chunks_md, file_path_md)]
    if not chunks:
        raise ValueError("No data files to chunknizer, put files in data/raw/vllm-0.10.1")
    model_chunks: list[MinimalSource] = [] 

    for c in chunks:
        model_chunks.append(MinimalSource(
                           chunk_content=c.page_content,
                           file_path=c.metadata["file_path"],
                           first_character_index=c.metadata["start_index"],
                           last_character_index=c.metadata["end_index"]
                           ))
    
    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    with open(f"{str(PROCESSED_DIR)}/chunks.json", "w") as f:    
        json.dump([model.model_dump() for model in model_chunks], f, indent=4)
    return (model_chunks, PROCESSED_DIR)
