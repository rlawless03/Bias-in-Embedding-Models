import gensim
import torch
import numpy as np
import os
from allennlp.fairness.bias_metrics import WordEmbeddingAssociationTest
from mlxtend.evaluate import permutation_test

# Load lists and filter out words not in embedding vocabulary
def load_words(path, embeddings):
    with open(path) as f:
        return [w.strip() for w in f.readlines() if w.strip() in embeddings]

# Convert lists to tensors 
def to_tensor(words, embeddings):
    vecs = np.array([embeddings[w] for w in words])
    return torch.tensor(vecs, dtype=torch.float32)

# permutation test method based on responsibly library documentation - https://docs.responsibly.ai/_modules/responsibly/we/weat.html
def calc_weat_pvalue(first_associations, second_associations, method): # method should equal 'exact' for small word sets and 'approximate' otherwise
    p_value = permutation_test(
        first_associations,
        second_associations,
        func=lambda x, y: sum(x) - sum(y),
        method=method,
        #num_rounds=1000000,
        seed=42
    )
    return p_value

def calc_association_scores(target_tensor, attr_tensor_A, attr_tensor_B):
    scores = []
    for word in target_tensor:
        mean_cos_A = torch.nn.functional.cosine_similarity(
            word.unsqueeze(0), attr_tensor_A
        ).mean().item()
        mean_cos_B = torch.nn.functional.cosine_similarity(
            word.unsqueeze(0), attr_tensor_B
        ).mean().item()
        scores.append(mean_cos_A - mean_cos_B)
    return scores

def generate_weat_results(filename_input, targets1, targets2, attributes1, attributes2):
    embedding_models = ["GoogleNews-vectors-negative300.bin"]
    weat = WordEmbeddingAssociationTest()
    results_filename = "output/" + filename_input
    # creating directories with file output - https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
    os.makedirs(os.path.dirname(results_filename), exist_ok=True) # overwrite results file
    with open(results_filename, "w") as f:
        pass

    for i in embedding_models:
        no_header = True # reset flags to suit majority of the embedding models
        is_binary = False
        print("Loading embeddings " + i)
        path = ("embeddings/" + i)

        if i == "numberbatch-en-17.06.txt":
            no_header = False
        if i == "GoogleNews-vectors-negative300.bin":
            no_header = False
            is_binary = True
        if i == "wiki-news-300d-1M-subword.vec":
            no_header = False
        if i == "wiki-news-300d-1M.vec":
            no_header = False
        embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            path, binary=is_binary, no_header=no_header
        )

        temp_targets1 = load_words(targets1, embeddings)
        temp_targets2 = load_words(targets2, embeddings)
        temp_attributes1 = load_words(attributes1, embeddings)
        temp_attributes2 = load_words(attributes2, embeddings)

        # Trim target lists to the same length if they are different sizes
        if len(temp_targets1) != len(temp_targets2):
            min_len = min(len(temp_targets1), len(temp_targets2))
            print("Lists are different lengths. Trimming both to " + str(min_len))
            temp_targets1 = temp_targets1[:min_len]
            temp_targets2 = temp_targets2[:min_len]
        print("Embeddings loaded")
        print(temp_targets1)
        print(temp_targets2)
        X = to_tensor(temp_targets1, embeddings)
        Y = to_tensor(temp_targets2, embeddings)
        A = to_tensor(temp_attributes1, embeddings)
        B = to_tensor(temp_attributes2, embeddings)

        score = float(weat(X, Y, A, B))

        # Compute per-word association scores for p-value calculation
        first_associations = calc_association_scores(X, A, B)
        second_associations = calc_association_scores(Y, A, B)

        p_value = calc_weat_pvalue(first_associations, second_associations, method='approximate')

        with open(results_filename, "a") as f:
            f.write(i + "\n\tWEAT: " + str(score) + "\n")
            f.write("\tP-Value: " + str(p_value) + "\n\n")
