FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir .

RUN rm -rf build & sphinx-build -b html ./source ./build

CMD ["python", "-m", "http.server", "8000", "--directory", "./build"]
