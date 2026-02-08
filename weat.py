import gensim
import torch
import numpy as np
import os
from allennlp.fairness.bias_metrics import WordEmbeddingAssociationTest

# Load embeddings using GloVe
print("Loading embeddings...")
embeddings = gensim.models.KeyedVectors.load_word2vec_format(
    "embeddings/glove.6B.50d.txt", binary=False, no_header=True
)
print("Embeddings loaded")

# Load lists and filter out words not in embedding vocabulary
def load_words(path):
    with open(path) as f:
        return [w.strip() for w in f.readlines() if w.strip() in embeddings]

targets_1 = load_words("targets/flowers.txt")
targets_2 = load_words("targets/insects.txt")
attributes_1 = load_words("attributes/pleasant.txt")
attributes_2 = load_words("attributes/unpleasant.txt")

# Convert lists to tensors 
def to_tensor(words):
    vecs = np.array([embeddings[w] for w in words])
    return torch.tensor(vecs, dtype=torch.float32)

X = to_tensor(targets_1)      # target group 1
Y = to_tensor(targets_2)      # target group 2
A = to_tensor(attributes_1)   # attribute group 1
B = to_tensor(attributes_2)   # attribute group 2

# Compute WEAT score 
weat = WordEmbeddingAssociationTest()
score = float(weat(X, Y, A, B))

print("\nWEAT score:", float(score))

# creating directories with file output - https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
filename = "output/weat_results.txt"
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, "w") as f:
    f.write("weat_score: " + str(score))
    