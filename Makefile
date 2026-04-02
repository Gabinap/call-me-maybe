ARGS ?=

install:
	uv sync

run:
	uv run -m src $(ARGS)

debug:
	@echo "   Starting debugger..."
	@echo "   Useful commands:"
	@echo "   n (next)       - Execute next line"
	@echo "   s (step)       - Step into function"
	@echo "   c (continue)   - Continue until next breakpoint"
	@echo "   p <var>        - Print variable"
	@echo "   l (list)       - Show source code"
	@echo "   q (quit)       - Quit debugger"
	@echo ""
	uv run python3 -m pdb src/main.py

lint:
	@uv run flake8 --exclude=.*,llm_sdk/*
	@uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude llm_sdk

lint-strict:
	@uv run flake8 --exclude=.*,llm_sdk/*
	@uv run mypy . --strict --exclude llm_sdk

clean:
	rm -rf .mypy_cache
	rm -rf .venv
	rm -rf *__pycache__*
	rm -rf .vscode
	rm -rf */*/output/

tests:
	uv run pytest tests/

.PHONY: install run debug lint clean tests
