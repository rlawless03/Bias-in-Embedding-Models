# Final Year Project - Detecting Bias in Text using AllenNLP

## Description

The objective is to measure bias in text using Implicit Association Test methods, such as the Word Embedding Association Test, through the use of AllenNLP’s built-in bias_metrics module. This data will allow the evaluation of bias in a range of embedding models.

Testing was performed on macOS Apple Silicon, through Docker.

## Prerequisites

- Docker Desktop installed

## Installation

### Create Docker image

Run the following command to create a Docker image in the current directory.

Make sure Dockerfile has been pulled.

```
docker build --platform=linux/amd64 -t <name> .
```

### Enter Interactive Shell

Once Docker image has been created, enter interactive shell

```
docker run -it --rm --platform linux/amd64 \
    -v <path>:/app \
    -w /app \
    --entrypoint /bin/bash  \
    <name>
```

### Run file

```
python weat.py
```
