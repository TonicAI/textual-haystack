.PHONY: test integration_tests lint format

test:
	uv run --group test pytest tests/unit_tests/

integration_tests:
	uv run --group test pytest tests/integration_tests/

lint:
	uv run --group lint ruff check src/ tests/
	uv run --group lint ruff format --diff src/ tests/

format:
	uv run --group lint ruff check --fix src/ tests/
	uv run --group lint ruff format src/ tests/
