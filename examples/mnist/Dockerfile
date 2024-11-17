FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev

COPY nanofed ./nanofed
COPY examples/mnist ./examples/mnist

CMD ["python", "examples/mnist/run_experiment.py"]
