VENV = .venv
UV_RUN = uv run python -m src

run:
	@$(UV_RUN)

install:
	@mkdir -p $$HOME/sgoinfre/.cache

	@if [ -d $$HOME/.cache ] && [ ! -L $$HOME/.cache ]; then \
	    cp -a $$HOME/.cache/. $$HOME/sgoinfre/.cache/; \
	    rm -rf $$HOME/.cache; \
	fi

	@if [ ! -e $$HOME/.cache ] && [ ! -L $$HOME/.cache ]; then \
	    ln -s $$HOME/sgoinfre/.cache $$HOME/.cache; \
	fi

	@test -d $(VENV) || uv sync

clean:
	@rm -rf src/__pycache__
	@rm -rf .vscode/
	@rm -rf data/processed
	@rm -rf data/output
	@rm -rf build/
	@rm -rf .mypy_cache/
	@clear

fclean: clean
	@rm -rf data
	@rm -rf .venv/

lint:
	@uv run flake8 .
	@uv run mypy .
	@echo "\033[32mEverything in the norm!!"

index:
	$(UV_RUN) index --max_chunk_size 2000

search:
	$(UV_RUN) search "How to configure a OpenIA server?" --k 5

search_dataset:
	$(UV_RUN) search_dataset data/datasets/UnansweredQuestions/dataset_code_public.json --k 5 data/output/search_results/

answer:
	$(UV_RUN) answer "How to configure a OpenIA server?" --k 5

answer_dataset:
	$(UV_RUN) answer_dataset data/output/search_results/dataset_code_public.json data/output

evaluate:
	$(UV_RUN) evaluate data/output/search_results/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json

deploy:
	uv run uvicorn src.api:app --reload &
	while ! curl -s http://127.0.0.1:8000/docs > /dev/null; do \
		sleep 1; done
	firefox http://127.0.0.1:8000/docs

killAPI:
	pkill -f uvicorn

help:
	@echo "\033[1;33mGeneral:\033[0m"
	@echo "  make run             - Run the RAG CLI"
	@echo "  make install         - Install project dependencies"
	@echo "  make clean           - Remove generated files and caches"
	@echo "  make fclean          - Remove generated files, caches and virtual environment"
	@echo "  make lint            - Run flake8 and mypy"
	@echo ""
	@echo "\033[1;33mRAG Pipeline:\033[0m"
	@echo "  make index           - Index the vLLM repository"
	@echo "  make search          - Search relevant chunks"
	@echo "  make search_dataset  - Search questions from a dataset"
	@echo "  make answer          - Generate an answer using RAG"
	@echo "  make answer_dataset  - Generate answers for a dataset"
	@echo "  make evaluate        - Evaluate search results"
	@echo ""
	@echo "\033[1;33mAPI:\033[0m"
	@echo "  make deploy          - Start the FastAPI server and open Swagger UI"
	@echo "  make killAPI         - Stop the Uvicorn server"

.PHONY: run install clean fclean lint index search search_dataset answer answer_dataset evaluate deploy killAPI help .DEFAULT

.DEFAULT:
	@echo "\033[31mError: Unknown command.\033[0m"
	@echo "Use \033[33mmake help\033[0m to see all available commands."