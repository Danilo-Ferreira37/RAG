
run:
	@uv run python -m src

install:
	uv sync

clean:
	@rm -rf src/__pycache__

fclean:
	@rm -rf src/__pycache__
	rm -rf .venv

index:
	uv run python -m src index --max_chunk_size 1500

search:
	uv run python -m src search --k 5

search_datase:
	uv run python -m src search_datase

answer:
	uv run python -m src answer

answer_datase:
	uv run python -m src answer_datase

evaluate:
	uv run python -m src evaluate