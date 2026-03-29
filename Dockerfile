FROM allennlp/allennlp:latest

RUN apt-get update && apt-get install -y git curl ca-certificates && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir gensim numpy mlxtend
