*This project has been created as part of the 42 curriculum by dosorio-.*

# RAG against the machine

A Retrieval-Augmented Generation (RAG) system that answers questions about the [vLLM](https://github.com/vllm-project/vllm) codebase (v0.10.1) and its documentation.

## Description

**Goal.** Given a natural-language question about vLLM (for example, *"What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?"*), the system finds the most relevant passages in the vLLM repository and uses them as context for a small language model to produce a grounded answer.

**Overview.** The project provides:

- An **indexer** that splits the vLLM source files (`.py`, `.md`, `.txt`) into overlapping chunks and builds a BM25 index.
- A **retriever** that returns the top-*k* chunks for a query, including the file path and the character range of each chunk.
- A **generator** that uses `Qwen/Qwen3-0.6B` to answer using only the retrieved context.
- An **evaluator** that measures retrieval quality with recall@k against a reference dataset.
- Two interfaces over the same pipeline: a **command-line interface** (built with [Fire](https://github.com/google/python-fire)) and a **REST API** (built with [FastAPI](https://fastapi.tiangolo.com/)) that runs on localhost.

## Instructions

### Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) (dependency and environment manager)
- `make`
- The vLLM v0.10.1 source tree placed at `data/raw/vllm-0.10.1/`
- The datasets placed under `datasets_public/public/` (`UnansweredQuestions/` and `AnsweredQuestions/`)

### Installation

```bash
make install        # runs `uv sync`
```

### Makefile commands

| Command | Description |
| --- | --- |
| `make install` | Install project dependencies |
| `make run` | Run the RAG CLI (`uv run python -m src`) |
| `make index` | Chunk the vLLM repository and build the BM25 index |
| `make search` | Retrieve the top-k chunks for a single query |
| `make search_dataset` | Retrieve the top-k chunks for every question in a dataset |
| `make answer` | Generate an answer for a single query |
| `make answer_dataset` | Generate answers for a dataset of retrieved results |
| `make evaluate` | Compute recall@k for a set of retrieval results |
| `make deploy` | Start the FastAPI server and open Swagger UI in Firefox |
| `make killAPI` | Stop the Uvicorn server |
| `make lint` | Run `flake8` and `mypy` |
| `make clean` | Remove caches, `data/processed` and `output` |
| `make fclean` | Same as `clean`, and also remove `.venv` |
| `make help` | List all available commands |

### CLI reference

All commands are exposed through `python -m src <command>`:

| Command | Arguments |
| --- | --- |
| `index` | `--max_chunk_size` (int, default and maximum `2000`) |
| `search` | `query`, `--k` |
| `search_dataset` | `dataset_path`, `--k`, `save_directory` |
| `answer` | `query`, `--k` |
| `answer_dataset` | `student_search_results_path`, `save_directory` |
| `evaluate` | `student_search_results_path`, `dataset_path` |

### REST API

Start the server with `make deploy`. It runs at `http://127.0.0.1:8000` and opens the interactive Swagger UI at `http://127.0.0.1:8000/docs`. Stop it with `make killAPI`.

| Method | Endpoint | Parameters | Description |
| --- | --- | --- | --- |
| `POST` | `/index` | `max_chunk_size` (default `2000`) | Build the chunks and the BM25 index |
| `GET` | `/search` | `query`, `k` | Return the top-k chunks as `MinimalSearchResults` |
| `GET` | `/answer` | `query`, `k` | Return an LLM-generated answer |
| `DELETE` | `/delete` | none | Delete `data/processed` and the BM25 index |

Errors raised by the pipeline are returned as HTTP `400` responses with the error message in `detail`.

## System architecture

```
                        INDEXING (make index)
┌───────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐
│ data/raw/         │──▶│ Chunking             │──▶│ BM25 index (bm25s)  │
│ vllm-0.10.1       │   │ (LangChain splitters)│   │ data/processed/     │
│ .py / .md / .txt  │   │ + chunks.json        │   │                     │
└───────────────────┘   └──────────────────────┘   └──────────┬──────────┘
                                                              │
                        QUERYING (search / answer)            │
┌───────────────────┐   ┌──────────────────────┐              │
│ Question          │──▶│ BM25 retrieval       │◀─────────────┘
│                   │   │ top-k, score > 0     │
└───────────────────┘   └──────────┬───────────┘
                                   │ chunks + file path + char range
                                   ▼
                        ┌──────────────────────┐   ┌─────────────────────┐
                        │ Prompt builder       │──▶│ Qwen3-0.6B          │──▶ Answer
                        │ (system + context)   │   │ (transformers)      │
                        └──────────────────────┘   └─────────────────────┘

                        EVALUATION (evaluate)
 retrieved sources  ──▶  same file path + IoU ≥ 0.05  ──▶  recall@k
```

### Components

| Component | File | Role |
| --- | --- | --- |
| Entry point | `src/__main__.py` | Starts the Fire CLI and converts errors into readable messages |
| Pipeline class | `src/cli.py` | `Rag` class: `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate` |
| Chunking | `src/chunking.py` | Reads the repository files, splits them and writes `chunks.json` |
| Data models | `src/models.py` | Pydantic models for sources, questions, results and answers |
| Evaluation | `src/evaluate.py` | IoU computation and recall@k |
| Dataset loading | `src/load.py` | Validates a dataset file and returns its questions |
| Exceptions | `src/execptions.py` | `RAGError`, raised when the index or context is missing |
| REST API | `src/api.py` | FastAPI wrapper around the `Rag` class |

### How the components interact

1. `index` calls the chunking module, which writes `data/processed/chunks.json`. The text of every chunk is then tokenized and indexed with `bm25s`, and the index is saved in `data/processed/`.
2. `search` and `answer` load the BM25 index, tokenize the query and retrieve the top-*k* chunk positions. These positions are mapped back to the `MinimalSource` objects stored in `chunks.json`.
3. For `answer`, the retrieved chunk texts are inserted into a chat prompt, and `Qwen/Qwen3-0.6B` generates the answer.
4. `search_dataset` and `answer_dataset` apply the same steps to every question of a dataset and save the results as JSON, so retrieval and generation can be run (and evaluated) separately.
5. `evaluate` compares the retrieved sources against the reference sources of an answered dataset.

All data exchanged between steps is validated with Pydantic models.

## Chunking strategy

Documents are segmented with LangChain's `RecursiveCharacterTextSplitter`, using a splitter adapted to each file type:

| File type | Splitter |
| --- | --- |
| `.py` | `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` |
| `.md` | `RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)` |
| `.txt` | `RecursiveCharacterTextSplitter` (default separators) |

- **Chunk size:** configurable with `--max_chunk_size`, with a default and maximum of **2000 characters**.
- **Overlap:** **10%** of the chunk size (200 characters with the default), so that information located at a chunk boundary is not lost.
- **Language-aware separators:** the Python and Markdown splitters prefer to cut at structural boundaries (classes, functions, headings, paragraphs) before falling back to finer separators. This keeps related code and documentation together.
- **Position tracking:** `add_start_index=True` records where each chunk begins in the original file. The end index is computed as `start_index + len(chunk) - 1`. Each chunk is stored as a `MinimalSource` with `file_path`, `first_character_index`, `last_character_index` and `chunk_content`, so every retrieved passage can be traced back to its exact location and compared to the reference sources.

## Retrieval method

Retrieval uses **BM25** (via the `bm25s` library), a lexical ranking function that scores each chunk by the frequency of the query terms in the chunk, weighted by how rare each term is across the corpus and normalized by chunk length.

1. The query is tokenized with `bm25s.tokenize`.
2. The index is queried for the **top-*k*** chunks, ranked by descending BM25 score.
3. Chunks with a score of `0` are discarded, because they share no terms with the query.
4. If no chunk remains, a `RAGError` is raised, indicating that the database does not have enough context for the query. In dataset mode, such a question is reported and skipped instead of stopping the run.

BM25 was chosen because the questions about vLLM contain many exact technical terms (endpoint names, flags, class names, file names) that lexical matching handles very well, and because it needs no embedding model or GPU.

## Answer generation

The retrieved chunks are concatenated into a prompt for `Qwen/Qwen3-0.6B`, applied through the tokenizer's chat template (with thinking mode disabled). The system prompt instructs the model to:

- answer **only** from the provided context;
- never invent facts, commands, parameters, file paths or configuration options;
- state clearly when the context is insufficient;
- answer directly and concisely, returning a single answer without repeating itself.

Generation is limited to `max_new_tokens=150`.

## Performance analysis

Retrieval quality is measured with **recall@k**. For each question in an answered dataset, the system checks whether at least one of the *k* retrieved chunks matches the **first** reference source of that question. A retrieved chunk matches when:

1. its `file_path` is identical to the reference file path, **and**
2. the **Intersection over Union (IoU)** between the character range of the retrieved chunk and the range of the reference source is **at least 0.05**.

```
recall@k = questions with at least one matching source in the top-k / questions evaluated
```

Run the evaluation with:

```bash
uv run python -m src evaluate <retrieved_results.json> <answered_dataset.json>
```

### Results

| Dataset | k | Recall@k | Questions evaluated |
| --- | --- | --- | --- |
| Docs | 5 | _TODO_ | _TODO_ |
| Code | 5 | _TODO_ | _TODO_ |

### Discussion

_TODO: comment on the recall obtained, how it changes with `k` and with the chunk size, which types of questions fail (for example, paraphrased questions with little lexical overlap), and the answer quality of the language model._

## Design decisions

- **BM25 instead of dense embeddings.** The corpus is technical and the questions contain exact identifiers, so lexical search is accurate, fast, deterministic and lightweight.
- **Language-aware chunking.** Splitting Python and Markdown along their own syntax keeps functions, classes and sections intact, which gives the model coherent context.
- **10% overlap with a 2000-character cap.** The overlap protects boundary information, and the cap keeps the prompt small enough for the model's context window.
- **Character indices stored with every chunk.** This makes each retrieved source verifiable and allows the IoU-based evaluation against the reference dataset.
- **Small local model (Qwen3-0.6B).** It runs on modest hardware without external API calls. A strict system prompt compensates for its limited capacity by forcing grounded and non-repetitive answers.
- **Retrieval and generation are separate steps.** `search_dataset` and `answer_dataset` write intermediate JSON files, so retrieval can be evaluated and tuned without paying for generation each time.
- **Pydantic validation everywhere.** Datasets, retrieval results and answers are validated on load and on save, so malformed files fail early with a clear error.
- **One pipeline, two interfaces.** The CLI and the REST API share the same `Rag` class, so both behave identically.
- **Centralized error handling.** A dedicated `RAGError`, plus handlers for invalid JSON, validation errors, missing files and invalid arguments, produces readable messages instead of stack traces.

## Challenges faced

- **Keeping chunks traceable to the original files.** Evaluation depends on exact character ranges. Solved by enabling `add_start_index` in the splitters and computing the end index for every chunk.
- **Queries without relevant context.** BM25 can return chunks with a score of zero. These are filtered out, and a `RAGError` is raised when nothing relevant is left. In dataset mode the question is skipped and reported, so the whole run does not fail.
- **Repetitive and invented answers from a small model.** A 0.6B model tends to repeat itself or to use outside knowledge. Solved with a strict system prompt, disabling thinking mode and limiting the number of generated tokens.
- **Choosing the chunk size.** Large chunks add noise and slow generation, while small chunks lose context. The 2000-character cap with 10% overlap was the compromise adopted.
- **Mixed file types.** Source code, Markdown and plain text have different structures, which led to one splitter per file type.
- **Running a cold pipeline.** Searching before indexing fails. This is handled with a clear message asking the user to run the `index` command first.

## Example usage

### 1. Install and index

```bash
make install
make index
```

Chunks are written to `data/processed/chunks.json` and the BM25 index is saved in `data/processed/`.

### 2. Search for a single query

```bash
uv run python -m src search "How to configure a OpenIA server?" --k 5
```

The top-5 sources are saved to `data/output/search_results.json`, in this format:

```json
{
    "question_id": "529d9fed-4ebc-45f6-ba9d-39647d5eeb95",
    "question": "How to configure a OpenIA server?",
    "retrieved_sources": [
        {
            "file_path": "data/raw/vllm-0.10.1/benchmarks/auto_tune/README.md",
            "first_character_index": 2800,
            "last_character_index": 4436,
            "chunk_content": "## How to Run\n\n1. **Configure**: Edit the script and set the variables..."
        }
    ]
}
```

### 3. Generate an answer for a single query

```bash
uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 5
```

### 4. Run a full dataset

```bash
# Retrieve the top-5 chunks for every question (writes output/search_results.json)
uv run python -m src search_dataset \
    datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
    --k 5 output

# Generate an answer for each question (writes output/StudentSearchResultsAndAnswer.json)
uv run python -m src answer_dataset output/search_results.json output
```

The dataset format is JSON. A question without an answer looks like this:

```json
{
  "rag_questions": [
    {
      "question_id": "17526382-7764-4120-b5e8-2d3726b8a4da",
      "question": "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?"
    }
  ]
}
```

A question with its reference answer and sources looks like this:

```json
{
  "question_id": "17526382-7764-4120-b5e8-2d3726b8a4da",
  "question": "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?",
  "answer": "The `/v1/load_lora_adapter` endpoint is used to dynamically load a LoRA adapter in vLLM. You send a POST request to this endpoint with the adapter's name and path in the JSON payload.",
  "sources": [
    {
      "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
      "first_character_index": 4695,
      "last_character_index": 6098
    }
  ]
}
```

### 5. Evaluate retrieval

```bash
uv run python -m src evaluate \
    output/search_results.json \
    datasets_public/public/AnsweredQuestions/dataset_docs_public.json
```

### 6. Use the REST API

```bash
make deploy      # starts the server and opens http://127.0.0.1:8000/docs

curl -X POST "http://127.0.0.1:8000/index?max_chunk_size=2000"
curl "http://127.0.0.1:8000/search?query=How%20to%20configure%20a%20OpenAI%20server&k=5"
curl "http://127.0.0.1:8000/answer?query=How%20to%20configure%20a%20OpenAI%20server&k=5"
curl -X DELETE "http://127.0.0.1:8000/delete"

make killAPI     # stops the server
```

## Project structure

```
.
├── Makefile
├── pyproject.toml
├── README.md
├── data/
│   ├── raw/vllm-0.10.1/        # vLLM source tree (input)
│   ├── processed/              # chunks.json and the BM25 index
│   └── output/                 # results of the `search` command
├── datasets_public/public/
│   ├── UnansweredQuestions/
│   └── AnsweredQuestions/
└── src/
    ├── __main__.py             # CLI entry point
    ├── api.py                  # FastAPI application
    ├── chunking.py             # document segmentation
    ├── cli.py                  # Rag class (full pipeline)
    ├── evaluate.py             # IoU and recall@k
    ├── execptions.py           # RAGError
    ├── load.py                 # dataset loading
    └── models.py               # Pydantic models
```

## Resources

