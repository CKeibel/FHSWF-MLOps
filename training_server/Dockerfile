# Base image with Python and scientific libraries
FROM python:3.11-slim

COPY . .

RUN pip install --no-cache-dir .

CMD ["python", "src/main.py"]
