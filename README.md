# MLOps

## Project Setup

### Pre-Commit

This project uses precommit for linting and formatting the code before it is actively committed to the project.
This helps to ensure code quality and code readability.

To install pre-commit, the following command can simply be run in the terminal.

```
# Install pre-commit
pip install pre-commit
```

In the root directory of the project there is a file `.pre-commit-config.yaml` in which the pre-commit is configured.

In order for the pre commit hooks to be executed automatically before each commit, the following command must be executed in the terminal.

```
# Install pre-commit hooks
pre-commit install
```

For more information visit the [pre-commit website](https://pre-commit.com/).

### Installation

The project is divided into two sub-projects, a fastapi backend and a gradio frontend.
Each of these sub-projects has its own `pyproject.toml` with which the project and its dependencies can be installed.

**Install backend**

1. To install the backend (fastapi restapi) switch the directory to *backend*
with `cd backend`.
2. Create a virtual environment to install the project dependencies in isolation.
This can be done by running `python -m venv .venv`.
3. To *activate* the environment run `source .venv/bin/activate` on mac/ linux or `./.venv/Scripts/activate` on windows.
4. To install all dependencies and to be ablte to develop and contribute to the project run `python -m pip install -e .'[dev]'`

**Install frontend**

1. To install the frontend (gradio) switch the directory to *backend*
with `cd frontend`.
2. Create a virtual environment to install the project dependencies in isolation.
This can be done by running `python -m venv .venv`.
3. To *activate* the environment run `source .venv/bin/activate` on mac/ linux or `./.venv/Scripts/activate` on windows.
4. To install all dependencies and to be ablte to develop and contribute to the project run `python -m pip install -e .'[dev]'`

# Project Structure

The project is structured into the following folder structure.



```
project/
├── backend/
│   ├── notebooks/ # Notebooks for experiments
│   ├── src/ # Backend implementation
│   ├── tests/ # Backend tests
│   ├── pyproject.toml # Dependencies
│   ├── README.md # Further instructions
│   └── ...
├── frontend/
│   ├── src/ # Frontend implementation
│   ├── tests/ # Frontend tests
│   ├── pyproject.toml # Dependencies
│   ├── README.md # Further instructions
│   └── ...
├── .pre-commit-config.yaml # pre-commit config
├── README.md # Project overview
└── ...
```
