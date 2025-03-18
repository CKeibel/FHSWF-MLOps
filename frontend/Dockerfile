FROM python:3.11-slim

RUN mkdir -p logs

COPY . .

RUN pip install --no-cache-dir .

CMD ["streamlit", "run", "src/main.py", "--server.port", "8501"]
