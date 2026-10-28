
run:
	@uv run python -m src

install:
	uv sync

clean:
	@rm -rf src/__pycache__
	@rm -rf .vscode
	@rm -rf data/processed
	@rm -rf output
	@rm -rf build

fclean:
	@rm -rf src/__pycache__
	@rm -rf .vscode
	@rm -rf data/processed
	@rm -rf output
	@rm -rf build
	@rm -rf .venv

index:
	uv run python -m src index --max_chunk_size 1500

search:
	uv run python -m src search "How to configure a OpenIA server?" --k 5

search_dataset:
	uv run python -m src search_dataset datasets_public/public/UnansweredQuestions/dataset_code_public.json --k 5 output

answer:
	uv run python -m src answer "How to configure a OpenIA server?" --k 5

answer_dataset:
	uv run python -m src answer_dataset output/StudentSearchResults.json output

evaluate:
	uv run python -m src evaluate