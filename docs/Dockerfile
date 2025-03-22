FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    pandoc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY . .

RUN pip install --no-cache-dir .

RUN rm -rf build & sphinx-build -b html ./source ./build

CMD ["python", "-m", "http.server", "8000", "--directory", "./build"]
