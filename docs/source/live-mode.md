# Live-Mode

## REST-API Backend

The Rest API currently consists of four endpoints.

- `/health` - To check the application health
- `predict` - To perform inference on a loaded model
- `set_model` - To deploy a new model by mlflow alias
- `model-info` - To receive information on the current deployed model
- `upload_data` - To upload new data

An initial training is performed during the start up of the API.

**IMPORTANT**: There will be a delay when starting for the first time)

To use one of the trained models via the backend, it must be assigned a `alias` in mlflow. The correct model can be loaded into the backend via the /set_model endpoint using its alias. The response will indicate which specific model has been loaded. If a model needs to be reverted, this can be done by changing the label to an older version.

## Monitoring

Monitoring is available through the /health and /model_info endpoints. The /health endpoint provides general information about the backend, while /model_info returns details about the currently loaded model. These endpoints can be used for future application cases as needed.
