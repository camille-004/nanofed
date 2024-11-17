.PHONY: install lint format type-check test

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check nanofed tests
	poetry run mypy nanofed

format:
	poetry run ruff format nanofed tests
	poetry run ruff check --fix nanofed tests

clean:
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +

build: clean
	poetry build

publish: build
	poetry publish

docker-build:
	docker build -t nanofed-dev -f Dockerfile.dev .

docker-test:
	docker run --rm nanofed-dev poetry run pytest

docker-example:
	docker-compose -f examples/mnist/docker-compose.yml up --build
