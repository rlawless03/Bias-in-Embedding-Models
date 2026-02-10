import gensim
import torch
import numpy as np
import os
from allennlp.fairness.bias_metrics import WordEmbeddingAssociationTest

# Load lists and filter out words not in embedding vocabulary
def load_words(path, embeddings):
    with open(path) as f:
        return [w.strip() for w in f.readlines() if w.strip() in embeddings]

# Convert lists to tensors 
def to_tensor(words, embeddings):
    vecs = np.array([embeddings[w] for w in words])
    return torch.tensor(vecs, dtype=torch.float32)

def generate_weat_results(results_filename, targets1, targets2, attributes1, attributes2):
    embedding_models = ["GoogleNews-vectors-negative300.bin", "numberbatch-en-17.06.txt", "glove.6B.50d.txt", "glove.6B.100d.txt", "glove.6B.200d.txt", "glove.6B.300d.txt", "glove.twitter.27B.25d.txt", "glove.twitter.27B.50d.txt", "glove.twitter.27B.100d.txt", "glove.twitter.27B.200d.txt"]
    weat = WordEmbeddingAssociationTest()
    results_filename = "output/" + results_filename
    
    # creating directories with file output - https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    with open(results_filename, "w") as f: # overwrite results file
        pass

    for i in embedding_models:
        no_header = True
        is_binary = False
        print("Loading embeddings " + i)
        path = ("embeddings/" + i)
    
        if i == "numberbatch-en-17.06.txt" or "GoogleNews-vectors-negative300.bin":
            no_header = False
    
        if i == "GoogleNews-vectors-negative300.bin":
            is_binary = True
        embeddings = gensim.models.KeyedVectors.load_word2vec_format(
        path, binary=is_binary, no_header=no_header    
    )
        temp_targets1 = load_words(targets1, embeddings)
        temp_targets2 = load_words(targets2, embeddings)
        temp_attributes1 = load_words(attributes1, embeddings)
        temp_attributes2 = load_words(attributes2, embeddings)
        print("Embeddings loaded")
        X = to_tensor(temp_targets1, embeddings)      # target group 1
        Y = to_tensor(temp_targets2, embeddings)      # target group 2
        A = to_tensor(temp_attributes1, embeddings)   # attribute group 1
        B = to_tensor(temp_attributes2, embeddings)   # attribute group 2
        
        score2 = float(weat(X, Y, A, B))
        
        with open(results_filename, "a") as f:
            f.write(i + " result: "  + str(score2) + "\n")
