from fastapi import FastAPI, HTTPException
from src.models import MinimalSearchResults
import shutil
from src.cli import Rag


app = FastAPI()
rag = Rag()


@app.post("/index")
def index(max_chunk_size: int = 2000) -> str:
    """
    Create the BM25 index from
    the repository files.

    Args:
        max_chunk_size: Maximum
            number of characters
            allowed per chunk.

    Returns:
        Index creation status.
    """
    try:
        rag.index(max_chunk_size)
        return "BM25 Indexing created!"
    except BaseException as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        ) from e


@app.get("/search")
def search(query: str, k: int) -> MinimalSearchResults:
    """
    Retrieve the most relevant
    chunks from the BM25 index
    based on the search query.

    Args:
        query: Text used to
            search the index.
        k: Maximum number of
            chunks to retrieve.

    Returns:
        Relevant repository chunks.
    """
    try:
        return rag._do_search(query, k)
    except BaseException as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        ) from e


@app.get("/answer")
def answer(query: str, k: int) -> str:
    """
    Generate an answer using
    the RAG pipeline, retrieving
    relevant chunks from the
    BM25 index as context.

    Args:
        query: Question to answer.
        k: Number of chunks used
            as context.

    Returns:
        LLM-generated answer.
    """
    try:
        return rag.answer(query, k)
    except BaseException as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        ) from e


@app.delete("/delete")
def delete() -> str:
    """
    Remove the processed data
    directory and its contents,
    including the BM25 index.

    Returns:
        Index deletion status.
    """
    shutil.rmtree("data/processed", ignore_errors=True)
    return "BM25 index deleted "
