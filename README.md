# Python Project Template 2024

![CI](https://github.com/camille-004/python_project/workflows/CI/badge.svg)

Welcome to the **Python Project Template 2024**. This will help get you started with the most modern Python tools and best practices.

## 🚀 Features

- **Depdendency Management** with [Poetry](https://python-poetry.org/)
- **Code Linting and Formatting** with [Ruff](https://github.com/astral-sh/ruff)
- **Type Checking** with [MyPy](https://mypy-lang.org/)
- **Automating Testing** with [Pytest](https://docs.pytest.org/en/8.2.x/)
- **Pre-commit Hooks** with [pre-commit](https://pre-commit.com/)
- **CI** with [GitHub Actions](https://github.com/features/actions)

## 📚 Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Makefile Commands](#makefile-commands)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Continuous Integration](#continuous-integration)
- [Contributing](#contributing)
- [License](#license)

## 🏁 Getting Started

### Prerequisites

Make sure you have the following installed:
- [Python 3.12+](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installing-with-pipx)

### Installation

1. **Clone the repository**

```sh
git clone https://github.com/camille-004/python_project.git
cd python_project
```

2. **Install dependencies**

```sh
make install
```

3. **Set up pre-commit hooks**

```sh
poetry run pre-commit install
```

### Adapting the Project

1. **Rename the project**

    Update the `pyproject.toml` file with your project's details:
    ```toml
    [tool.poetry]
    name = "your_project_name"
    version = "0.1.0"
    description = "Your project description"
    authors = ["Your Name <your.email@example.com>"]
    readme = "README.md"
    packages = [{include = "your_project_name"}]
    ```

    Then, rename the source code directory from `python_project` to the value of `your_project_name`.

2. **Customize or replace the `README`**

3. **Start developing**

    Add your source code to the `your_project_name/` directory and your tests to the `tests/` directory.

## 📂 Project Structure
```
.
├── .github
│   └── workflows
│       └── ci.yml
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── python_project
│   └── __init__.py
└── tests
    ├── __init__.py
    └── test_sample.py
```

## 🛠️ Makefile Commands

This project uses a `Makefile` to simplify various tasks:

- **Installing depedencies**
    ```sh
    make install
    ```

- **Running linters**
    ```sh
    make lint
    ```

- **Formatting code**
    ```sh
    make format
    ```

- **Type-checking the code**
    ```sh
    make type-check
    ```

- **Running tests**
    ```sh
    make test
    ```

## 📝 Pre-commit Hooks

Pre-commit hooks are configured to run the following checks before every commit:
- **Lint**: `make lint`
- **Format**: `make format`
- **Type Check**: `make type-check`

## ⚙️ Continuous Integration

This project uses GitHub Actions for Continuous Integration. Check `.github/workflows/ci.yml` for the steps in how the workflow is defined:

- **Install dependencies**
- **Run linters**
- **Type-check the code**
- **Run tests**

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or new features.

1. **Fork the repository**
2. **Create a new branch**
3. **Make your changes**
4. **Submit a pull request**

## 📄 License

This project is licensed under the GNU General Public License. See the [LICENSE](LICENSE) file for details.

---

Happy coding. 🚀
