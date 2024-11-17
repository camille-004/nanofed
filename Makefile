.PHONY: install dev test lint format clean build docs docker-build docker-dev docker-test docker-lint run-example help

help:
	@echo "Available commands:"
	@echo "Local Development:"
	@echo "  install      - Install dependencies using poetry"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linters (ruff, mypy)"
	@echo "  format      - Format code with ruff"
	@echo "  clean       - Clean up cache and build files"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build    - Build Docker development image"
	@echo "  docker-dev      - Start development shell in Docker"
	@echo "  docker-test     - Run tests in Docker"
	@echo "  docker-lint     - Run linters in Docker"
	@echo "  run-example     - Run MNIST example in Docker"
	@echo ""
	@echo "Package Management:"
	@echo "  build       - Build package distribution"
	@echo "  publish     - Publish package to PyPI"

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check .
	poetry run mypy nanofed

format:
	poetry run ruff format .
	poetry run ruff check --fix .

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

# Docker commands
docker-build:
	docker-compose build

docker-dev:
	docker-compose run --rm dev

docker-test:
	docker-compose run --rm dev poetry run pytest

docker-lint:
	docker-compose run --rm dev poetry run ruff check .
	docker-compose run --rm dev poetry run mypy nanofed

docker-clean:
	docker-compose down -v
	docker system prune -f

run-example:
	docker-compose run --rm dev python examples/mnist/run_experiment.py

setup-pre-commit:
	poetry run pre-commit install
	poetry run pre-commit run --all-files

ci-check: lint test build

.DEFAULT_GOAL := help
