# MLOps

## Project Setup

### Pre-Commit

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

### Installation

The project is divided into two sub-projects, a fastapi backend and a gradio frontend.
Each of these sub-projects has its own `pyproject.toml` with which the project and its dependencies can be installed.


*There is also a `requirements.txt` in the root directory to install all dependencies into a single isolated environment. The separation of `pyproject.toml` is necessary for the containerisation of the two components of the application*

**Single installation**

1. Create a virtual environment to install the project dependencies in isolation.
This can be done by running `python -m venv .venv`.
2. To *activate* the environment run `source .venv/bin/activate` on mac/ linux or `./.venv/Scripts/activate` on windows.
3. To install all dependencies and to be able to develop and contribute to the project run `python -m pip install -r requirements.txt`

**Install backend**

1. To install the backend (fastapi restapi) switch the directory to *backend*
with `cd backend`.
2. Create a virtual environment to install the project dependencies in isolation.
This can be done by running `python -m venv .venv`.
3. To *activate* the environment run `source .venv/bin/activate` on mac/ linux or `./.venv/Scripts/activate` on windows.
4. To install all dependencies and to be able to develop and contribute to the project run `python -m pip install -e .'[dev]'`

**Install frontend**

1. To install the frontend (gradio) switch the directory to *backend*
with `cd frontend`.
2. Create a virtual environment to install the project dependencies in isolation.
This can be done by running `python -m venv .venv`.
3. To *activate* the environment run `source .venv/bin/activate` on mac/ linux or `./.venv/Scripts/activate` on windows.
4. To install all dependencies and to be able to develop and contribute to the project run `python -m pip install -e .'[dev]'`

# Project Structure

The project is structured into the following folder structure.

```
project/
├── backend/
│   ├── notebooks/ # Notebooks for experiments
│   ├── src/ # Backend implementation
│       │── training/ # Implementation of Trainings
            │── data # origin Data for training
│           │── models/ # Base Model and Models for Experiments
│   ├── tests/ # Backend tests
│   ├── Dockerfile # Containerization
│   ├── pyproject.toml # Dependencies
│   ├── README.md # Further instructions
│   └── settings.yaml # Setting File
├── frontend/
│   ├── src/ # Frontend implementation
│   ├── tests/ # Frontend tests
│   ├── Dockerfile # Containerization
│   ├── pyproject.toml # Dependencies
│   ├── README.md # Further instructions
│   └── ...
├── .pre-commit-config.yaml # pre-commit config
├── README.md # Project overview
└── ...
```
# Docker containerization

Both sub-projects include a related `Dockerfile`.
You can build the image by running the following commands:

**backend**
```
# Switch to backend directory
cd backend
# Build
docker build -t mlops-backend .
# Run on port 8080
docker run -d -p 8080:8080 --restart unless-stopped --name mlops-backend mlops-backend
```

**frontend**
```
# Switch to frontend directory
cd frontend
# Build
docker build -t mlops-frontend .
# Run on port 8501
docker run -d -p 8501:8501 --restart unless-stopped --name mlops-frontend mlops-frontend
# Open Frontend for Inference via http://127.0.0.1:8501
```

# Training

## Adding New Experiments
To train a new model, it must be added under models in settings.yaml. The model must implement at least the functions defined in base.py. The existing models RandomForest and XGBoost serve as examples.

## Versioning of Training Data
To ensure versioning of training data, the data is stored as an artifact for each experiment. Additionally, all relevant metrics for the model are logged in MLflow. Each training dataset is evaluated as described, and the corresponding information is stored as artifacts in the MLflow backend.

## Automated Analysis of New Data

### Goal

The automated analysis checks whether newly incoming data still matches the data originally used for model training. This ensures that the model continues to make reliable predictions in the future. Changes in data structure or distributions can be detected and addressed early on.

The analysis runs automatically, creates an HTML report, and saves all results as artifacts in MLflow.

### What is specifically done?

#### 1. Comparison of old and new data

- Have columns been removed or added?
- Have data types changed?
- Are there changes in the value range for numerical variables?
- For categorical variables, have categories been added or removed?

#### 2. Examination of distributions

- Target variable ("income"): Has the distribution changed?
- For all features: Graphical comparison of the distribution between old and new datasets

#### 3. Calculation of correlations

- Which features correlate strongly with the target variable?
- Are there strong correlations between individual features?

#### 4. Further data quality checks and handling instructions

- Check for missing values (NaN or "?")
- Identification of duplicates
- Check for highly imbalanced features (e.g., features where a single value accounts for over 95% of the data)

### Next steps

The code developed here allows for quick and systematic identification of potential problem areas in the data in the future. It also provides concrete suggestions on which cleanups should be performed before retraining the model if necessary.


## Data Preparation
Data preparation is performed using an sklearn pipeline as seen in the example models. Each model can use a “customprocessor” and a “preprocessor” to prepare the data. The “customprocessor” adjusts the fields according to the findings from the data evaluation. The preprocessor is used to encode the categorical data “one-hot”, for example, or to use a MinMaxScaler for numerical values. This pipeline is stored together with the model in the MLflow backend after training and is used for inference. This ensures that all data is processed consistently within the model.

## Hyperparameter Tuning
For the example models, hyperparameter tuning is performed. The best result from this Optuna study is then stored in MLflow Models for each model. The value ranges for hyperparameter tuning are defined in the respective model class.

## Labeling der Modelle
The latest model is labeled as newest. Note that multiple experiments may result in multiple models labeled as newest. It is recommended to use labels such as "Production" and "Staging" to maintain clarity when accessing models via the backend. This labeling must be performed manually in the MLflow backend to ensure a clear transition between staging and production.

## Adding new data
New training data can be added via Fileupload to the backend. When new training data is added, retraining is performed, and the training data, model, and corresponding results are stored. If the model should be deployed to production, it must be labeled accordingly and loaded into the backend via the /set_model endpoint. The settings.yaml file allows specifying the general path for training data and the path to the original training data. During training, a comparison between the current and original training data is performed and documented in report.html in the artifacts folder. Additional visualizations of the training dataset can also be found in the artifacts folder. The training of new models occurs in the background and does not block the API

# Live-Betrieb

## REST-API Backend
To use one of the trained models via the backend, it must be assigned a label. The correct model can be loaded into the backend via the /set_model endpoint using its alias. The response will indicate which specific model has been loaded. If a model needs to be reverted, this can be done by changing the label to an older version.

## Monitoring

Monitoring is available through the /health and /model_info endpoints. The /health endpoint provides general information about the backend, while /model_info returns details about the currently loaded model. These endpoints can be used for future application cases as needed.

## Frontend



# Deployment

## Pre-Commit

## CI CD Pipeline

## Unit Tests und Integration Tests


These Project use MLFlow as a Backend for managing the lifecycle for models. After each upload from new data over the endpoint /upload_data a new training will be started.

hinzufügen von neuen modellen


BEst Model warum nicht gesetzt?
Nur New gesetzt
Datenversionierung in MLFlow Artifacs Daten aus Ordner über Runid holen
Monitoring über Healthcheck
Neue Daten fügt zu neu Training
Setzen der MOdelle findet über Alias statt
Machen keine
Hyperparametertraining
Nach dem Hochladen von Daten wird neiu trainiert und Modell in Experiments gesichert. Bestes Model nach optuna Tuning wird als Model gespeichert und kann gelabelt werden und dann über das Backend angesporchen werden
Bei Training wird API nicht blockiert
