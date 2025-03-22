# Environment / Project Setup

## Pre-Commit

This project uses **pre-commit** for linting and formatting the code before it is actively committed to the codebase.
This helps to ensure code quality and code readability.

To install pre-commit, the following command can simply be run in the terminal.

```
# Install pre-commit
pip install pre-commit
```

In the root directory of the project there is a `.pre-commit-config.yaml` file in which the pre-commit hooks are configured.

In order for the pre-commit hooks to be executed automatically before each commit, the following command must be executed in the terminal.

```
# Install pre-commit hooks
pre-commit install
```

For more information visit the [pre-commit website](https://pre-commit.com/).

## Installation

The project is divided into three sub-projects, a fastapi backend and a streamlit frontend and a sphinx documentation. Besides that a Dockerfile for the *MLFlow Server* is implemented.
Each of these sub-projects has its own `pyproject.toml` with which the project and its dependencies can be installed.


*There is also a `requirements.txt` in the root directory to install all dependencies into a single isolated environment. The separation of `pyproject.toml` is necessary for the containerisation of the two components of the application*

## Single installation

1. Create a virtual environment to install the project dependencies in isolation.
This can be done by running `python -m venv .venv`.
2. To *activate* the environment run `source .venv/bin/activate` on mac/ linux or `./.venv/Scripts/activate` on windows.
3. To install all dependencies and to be able to develop and contribute to the project run `python -m pip install -r requirements.txt`

### Install backend

1. To install the backend (fastapi restapi) switch the directory to *backend*
with `cd backend`.
2. Create a virtual environment to install the project dependencies in isolation.
This can be done by running `python -m venv .venv`.
3. To *activate* the environment run `source .venv/bin/activate` on mac/ linux or `./.venv/Scripts/activate` on windows.
4. To install all dependencies and to be able to develop and contribute to the project run `python -m pip install -e .'[dev]'`

### Install frontend

1. To install the frontend (gradio) switch the directory to *backend*
with `cd frontend`.
2. Create a virtual environment to install the project dependencies in isolation.
This can be done by running `python -m venv .venv`.
3. To *activate* the environment run `source .venv/bin/activate` on mac/ linux or `./.venv/Scripts/activate` on windows.
4. To install all dependencies and to be able to develop and contribute to the project run `python -m pip install -e .'[dev]'`

### Build the docs

To build the documentation locally, use `cd docs` to change to the directory. The following command needs to be executed to build the docs:

```
rm -rf build & sphinx-build -b html ./source ./build
```

To start the documentation locally, the following must be run:

```
python -m http.server 8080 --directory ./build
```

## Docker

All sub-projects include a related `Dockerfile`.
You can build the image by running the following commands:

### backend
```
# Switch to backend directory
cd backend
# Build
docker build -t mlops-backend .
# Run on port 8080
docker run -d -p 8080:8080 --restart unless-stopped --name mlops-backend mlops-backend
```

### frontend
```
# Switch to frontend directory
cd frontend
# Build
docker build -t mlops-frontend .
# Run on port 8501
docker run -d -p 8501:8501 --restart unless-stopped --name mlops-frontend mlops-frontend
# Open Frontend for Inference via http://127.0.0.1:8501
````

### docs
```
# Switch to frontend directory
cd docs
# Build
docker build -t docs .
# Run on port 8000
docker run -d -p 8000:8000  --name docs docs
# Open Frontend for Inference via http://127.0.0.1:8000
```

## Docker Compose

This `docker-compose.yaml` file defines a multi-container application with four services: mlflow, backend, frontend, and docs. It also uses a shared volume named `mlflow_data` to persist data.

The services are built from their respective Dockerfiles and directories.

The services communicate with each other using Docker's internal networking (e.g., http://mlflow, http://backend, etc.).

The shared volume (`mlflow_data`) ensures that experiment data is persisted and accessible across services (`mlflow`, `backend`).

## Project Structure

The project is structured into the following folder structure.

```
project/
├── backend/
│   ├── notebooks/ # Notebooks for experiments
│   ├── src/ # Backend implementation
│   │    │── training/ # Implementation of Trainings
│   │         │── data # origin Data for training
│   │         │── models/ # Base Model and Models for Experiments
│   ├── tests/ # Backend tests
│   ├── Dockerfile # Containerization
│   ├── pyproject.toml # Dependencies
│   └── settings.yaml # Setting File
├── frontend/
│   ├── src/ # Frontend implementation
│   ├── tests/ # Frontend tests
│   ├── Dockerfile # Containerization
│   ├── pyproject.toml # Dependencies
│   └── ...
├── docs/
│   ├── build/ # Frontend implementation
│   ├── source/ # Frontend tests
│   ├── Dockerfile # Containerization
│   └── pyproject.toml # Dependencies
├── .pre-commit-config.yaml # pre-commit config
├── README.md # Project overview
└── ...
```

## VS Code (Debug) Run Config

In addition to the local start run configurations are available for the `frontend` and `backend`, which can be selected at the top left in vs code next to the green play button.

It is **important** to have installed the dependencies beforehand and to have selected the correct python interpreter in the IDE.

The respective run config has set the `current working directory` to the project folder so that the `dynconf settings` can be read correctly.

The mlflow must be started manually in the backend with `mlflow ui`.

```
# launch.json
{
    "version": "0.2.0",
    "configurations": [

        {
            "name": "Backend",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "cwd": "${workspaceFolder}/backend",
            "args": [
                "src.main:app",
                "--reload"
            ],
            "jinja": true
        },
        {
            "name": "Frontend",
            "type": "debugpy",
            "request": "launch",
            "module": "streamlit",
            "cwd": "${workspaceFolder}/frontend",
            "args": [
                "run",
                "src/main.py"
            ],
            "console": "integratedTerminal"
        }
    ]
}
```
