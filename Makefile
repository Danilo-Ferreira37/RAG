VENV = .venv
UV_RUN = uv run python -m src



run:
	@$(UV_RUN) answer "How to configure a OpenIA server?" --k 5

install:
	uv sync

clean:
	@rm -rf src/__pycache__
	@rm -rf .vscode/
	@rm -rf data/processed
	@rm -rf output
	@rm -rf build/
	@rm -rf .mypy_cache/
	@clear

fclean:
	@rm -rf src/__pycache__
	@rm -rf .vscode
	@rm -rf data/processed
	@rm -rf output/
	@rm -rf build/
	@rm -rf .mypy_cache/
	@rm -rf .venv
	@clear

lint:

	@uv run flake8 .
	@uv run mypy .
	@echo "\033[32mEverything in the norm!!"

index:
	$(UV_RUN) index --max_chunk_size 2000

search:
	$(UV_RUN) search "How to configure a OpenIA server?" --k 5

search_dataset:
	$(UV_RUN) search_dataset datasets_public/public/UnansweredQuestions/dataset_code_public.json --k 5 output

answer:
	$(UV_RUN) answer "How to configure a OpenIA server?" --k 5

answer_dataset:
	$(UV_RUN) answer_dataset output/StudentSearchResultsdocs.json output

evaluate_code:
	$(UV_RUN) evaluate output/StudentSearchResultscode.json datasets_public/public/AnsweredQuestions/dataset_code_public.json

evaluate_docs:
	$(UV_RUN) evaluate output/StudentSearchResultsdocs.json datasets_public/public/AnsweredQuestions/dataset_docs_public.json