FROM python:3.11-slim

COPY . .

RUN pip install --no-cache-dir .


CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "80"]
