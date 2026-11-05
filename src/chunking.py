from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from pathlib import Path
from src.models import MinimalSource
from typing import Any
import json
from tqdm import tqdm


DATA_DIR = Path("data/raw/vllm-0.10.1")


def get_chunking_of_python_files(
        max_chunk_size: int,
        overlap: int,
        files_content: list[str],
        files_path: list[dict[str, Any]]
) -> list[Document]:
    """
    Split Python source files into overlapping text chunks.

    Uses a Python-specific text splitter to preserve the
    structure of the source code and records the character
    indices of each chunk within its original file.
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        add_start_index=True
    )

    chunks = splitter.create_documents(files_content, metadatas=files_path)

    for chunk in chunks:
        start_index = chunk.metadata.get("start_index")

        if not isinstance(start_index, int):
            raise ValueError("Invalid start_index")

        chunk.metadata["end_index"] = (
            len(chunk.page_content) + start_index - 1
        )

    return chunks


def get_chunking_of_txt_files(
        max_chunk_size: int,
        overlap: int,
        files_content: list[str],
        files_path: list[dict[str, Any]]
) -> list[Document]:
    """
    Split plain text files into overlapping text chunks.

    Uses a recursive text splitter to divide the documents
    according to the specified chunk size and overlap.

    Records the character indices of each chunk.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        add_start_index=True
    )

    chunks = splitter.create_documents(files_content, metadatas=files_path)

    for chunk in chunks:
        start_index = chunk.metadata.get("start_index")

        if not isinstance(start_index, int):
            raise ValueError("Invalid start_index")

        chunk.metadata["end_index"] = (
            len(chunk.page_content) + start_index - 1
        )

    return chunks


def get_chunking_of_md_files(
        max_chunk_size: int,
        overlap: int,
        files_content: list[str],
        files_path: list[dict[str, Any]]
) -> list[Document]:
    """
    Split Markdown files into overlapping text chunks.

    Uses a Markdown-specific text splitter to preserve
    document structure and records the character indices
    of each chunk within its original file.
    """

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        add_start_index=True
    )

    chunks = splitter.create_documents(files_content, metadatas=files_path)

    for chunk in chunks:
        start_index = chunk.metadata.get("start_index")

        if not isinstance(start_index, int):
            raise ValueError("Invalid start_index")

        chunk.metadata["end_index"] = (
            len(chunk.page_content) + start_index - 1
        )

    return chunks


def get_chunks_and_put_in_file(
        max_chunk_size: int,
        overlap: int,
        PROCESSED_DIR: Path
) -> list[MinimalSource]:
    """
    Process source files and save their chunks to a JSON file.

    Collects Python, Markdown, and plain text files from
    the source directory and splits them into chunks.

    Converts the chunks into MinimalSource objects, saves
    them to the processed directory, and returns the list.
    """
    chunks_py: list[str] = []
    chunks_md: list[str] = []
    chunks_txt: list[str] = []

    file_path_py: list[dict[str, Any]] = []
    file_path_md: list[dict[str, Any]] = []
    file_path_txt: list[dict[str, Any]] = []

    for file_path in tqdm(
        DATA_DIR.rglob("*"),
        desc="Open all files in data/raw/"
    ):

        if file_path.is_file() and file_path.suffix == ".py":
            with open(file_path, encoding="utf-8") as f:
                chunks_py.append(f.read())
                file_path_py.append({"file_path": str(file_path)})

        elif file_path.is_file() and file_path.suffix == ".md":
            with open(file_path, encoding="utf-8") as f:
                chunks_md.append(f.read())
                file_path_md.append({"file_path": str(file_path)})

        elif file_path.is_file() and file_path.suffix == ".txt":
            with open(file_path, encoding="utf-8") as f:
                chunks_txt.append(f.read())
                file_path_txt.append({"file_path": str(file_path)})

    chunks = [
        *get_chunking_of_txt_files(
            max_chunk_size, overlap, chunks_txt, file_path_txt
        ),
        *get_chunking_of_md_files(
            max_chunk_size, overlap, chunks_md, file_path_md
        ),
        *get_chunking_of_python_files(
            max_chunk_size, overlap, chunks_py, file_path_py
        )
    ]

    if not chunks:
        raise ValueError(
            "No data files to chunknizer, "
            "put files in data/raw/vllm-0.10.1 with "
            "suffixes '.py', '.md', '.txt'"
        )

    model_chunks: list[MinimalSource] = []
    for c in chunks:
        model_chunks.append(
            MinimalSource(
                chunk_content=c.page_content,
                file_path=c.metadata["file_path"],
                first_character_index=c.metadata["start_index"],
                last_character_index=c.metadata["end_index"]
            ))

    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    with open(PROCESSED_DIR / "chunks.json",
              "w",
              encoding="utf-8") as f:
        json.dump(
            [model.model_dump() for model in model_chunks],
            f,
            indent=4
        )

    return model_chunks
