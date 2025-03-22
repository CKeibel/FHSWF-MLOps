
# Deployment: Azure Multi-Container Application with GitHub Actions

This deployment involves hosting a multi-container application (FastAPI, Streamlit, MLflow, and Sphinx) on **Azure App Service** using **Docker Compose**. The setup includes **Azure Container Registry (ACR)** for storing Docker images, a **Nginx reverse proxy** for routing traffic, and **GitHub Actions** for CI/CD automation.

---

## **1. GitHub Actions for CI/CD**
GitHub Actions automates the build, push, and deployment process for your Docker images and application.

### **Pipeline Workflow**
1. **Code Checkout**: The repository code is checked out.
2. **Login to Azure Container Registry (ACR)**:
   - The pipeline uses a Service Principal (`AZURE_CREDENTIALS`) and ACR credentials (`REGISTRY_LOGIN_SERVER`, `REGISTRY_USERNAME`, `REGISTRY_PASSWORD`) to authenticate.
3. **Build and Push Docker Images**:
   - Each service image (FastAPI, Streamlit, MLflow, Sphinx, and Nginx) is built using its respective Dockerfile.
   - The images are pushed to ACR.
4. **Deployment to Azure App Service**:
   - The updated `docker-compose.prod.yml` file is used to deploy the multi-container application on Azure.

---

## **2. Azure Container Registry (ACR)**
ACR serves as the central storage location for all Docker images. Each image is tagged (e.g., `latest`) and retrieved by Azure App Service during deployment.

### Images in ACR:
- `mlops.azurecr.io/backend:latest`
- `mlops.azurecr.io/frontend:latest`
- `mlops.azurecr.io/mlflow:latest`
- `mlops.azurecr.io/docs:latest`
- `mlops.azurecr.io/nginx-custom:latest`

---

## **3. Azure App Service for Multi-Container Deployment**
Azure App Service hosts the multi-container application using Docker Compose. It orchestrates multiple containers under a single web app.

### Key Configurations in Azure App Service:
1. **Docker Compose File**:
   - The `docker-compose.prod.yml` file defines all services (FastAPI, Streamlit, MLflow, Sphinx) and the Nginx reverse proxy.
   - Example:
     ```
     version: '3.8'

        services:

        nginx:
            image: mlops.azurecr.io/nginx:latest
            ports:
                - "80:80"

        mlflow:
            image: mlops.azurecr.io/mlflow:latest
            ports:
                - "5000:5000"
            volumes:
            - mlflow_data:/mlflow
            restart: unless-stopped


        backend:
            image: mlops.azurecr.io/backend:latest
            ports:
                - "8080:8080"
            environment:
                - DYNACONF_MLFLOW__TRACKING_URI=http://mlflow:5000
            volumes:
                - mlflow_data:/app/mlruns
            depends_on:
                - mlflow
            restart: unless-stopped

        frontend:
            image: mlops.azurecr.io/frontend:latest
            ports:
                - "8501:8501"
            environment:
                - DYNACONF_BACKEND_URL=http://backend:8080/predict
            depends_on:
                - backend
            restart: unless-stopped

        docs:
            image: mlops.azurecr.io/docs:latest
            ports:
                - "8000:8000"
            restart: unless-stopped

        volumes:
        mlflow_data:

     ```

2. **Environment Variables**:
   - Set the variable `WEBSITES_PORT=80` to route traffic correctly through Nginx.

3. **Ports**:
   - All services are mapped through the reverse proxy on port 80.

---

## **4. Nginx as Reverse Proxy**
The Nginx service acts as the entry point for all requests and routes them to the appropriate services based on URL paths.

### Nginx Configuration (`nginx.conf`):

```
server {
    listen 80;

    location /api/ {
        proxy_pass http://backend:8080/;
        proxy_set_header Host $host;
    }

    location /frontend/ {
        proxy_pass http://frontend:8501/;
        proxy_set_header Host $host;
    }

    location /mlflow/ {
        proxy_pass http://mlflow:5000/;
        proxy_set_header Host $host;
    }

    location /docs/ {
        proxy_pass http://docs:8000/;
        proxy_set_header Host $host;
    }
}

```


This configuration ensures that requests are routed as follows:
- `/api/` → FastAPI service on port 8080.
- `/frontend/` → Streamlit service on port 8501.
- `/mlflow/` → MLflow service on port 5000.
- `/docs/` → Sphinx service on port 8000.

---

## **5. Accessing the Application**
After deployment, the application can be accessed via the Azure Web App URL:

| Service       | URL                                      |
|---------------|------------------------------------------|
| FastAPI       | `hhttps://mlops-adult-income-ehgdgtbpcwcqh5c6.westeurope-01.azurewebsites.net/api/` |
| Streamlit     | `https://mlops-adult-income-ehgdgtbpcwcqh5c6.westeurope-01.azurewebsites.net/frontend/` |
| MLflow        | `https://mlops-adult-income-ehgdgtbpcwcqh5c6.westeurope-01.azurewebsites.net/mlflow/` |
| Sphinx Docs   | `https://mlops-adult-income-ehgdgtbpcwcqh5c6.westeurope-01.azurewebsites.net/docs/` |

All traffic flows through Nginx on port 80.

---

## Summary of Deployment Components

### **GitHub Actions**
Automates CI/CD processes including building Docker images, pushing them to ACR, and deploying the application.

### **Azure Container Registry (ACR)**
Stores all Docker images securely with tags for easy retrieval during deployment.

### **Azure App Service**
Hosts the multi-container application using Docker Compose with environment variables and port configurations.

### **Nginx Reverse Proxy**
Routes incoming requests based on URL paths to the appropriate container services.

This setup provides a scalable and efficient solution for deploying multi-container applications on Azure while ensuring all services are accessible via a single entry point.
