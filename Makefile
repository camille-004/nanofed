.PHONY: install dev test lint format clean build docs docker-build docker-dev docker-test docker-lint run-example help

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check nanofed/ tests/
	poetry run mypy nanofed

format:
	poetry run ruff format nanofed/ tests/
	poetry run ruff check --fix nanofed/ tests/

clean:
	# Clean Python cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	# Clean build artifacts
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	# Clean example artifacts
	find examples -type d -name "runs" -exec rm -rf {} +
	find . -type f -name "*.pt" -delete
	find . -type f -name "*.log" -delete

# Package building and publishing
build: clean
	poetry build

publish: build
	poetry publish

docs:
	poetry run sphinx-build -b html docs/source docs/build/html

docs-serve: docs
	@echo "Serving documentation at http://localhost:8000"
	@cd docs/build/html && python -m http.server 8000

setup-pre-commit:
	poetry run pre-commit install
	poetry run pre-commit run --all-files
