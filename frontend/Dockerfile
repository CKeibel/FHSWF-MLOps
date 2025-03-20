FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src ./src
COPY src/main.py .

RUN pip install --no-cache-dir .

CMD ["streamlit", "run", "main.py", "--server.port", "8501"]
