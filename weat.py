import gensim
import torch
import numpy as np
import os
from allennlp.fairness.bias_metrics import WordEmbeddingAssociationTest
from scipy import stats

# Load lists and filter out words not in embedding vocabulary
def load_words(path, embeddings):
    with open(path) as f:
        return [w.strip() for w in f.readlines() if w.strip() in embeddings]

# Convert lists to tensors 
def to_tensor(words, embeddings):
    vecs = np.array([embeddings[w] for w in words])
    return torch.tensor(vecs, dtype=torch.float32)

def generate_pair_values(target_tensor, targets, attribute_tensor, attributes, file_name, pair_list):
    counter_t = 0
    # Find cosine similarity values for target-attribute pair which will be used to generate p-values
    for a in target_tensor:
        counter_a = 0
        for b in attribute_tensor:
            pair_score = torch.nn.functional.cosine_similarity(a.unsqueeze(0),b.unsqueeze(0)) # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html
            pair_list.append(float(pair_score))
            with open(file_name, "a") as f:
                f.write("\t" + targets[counter_t] + "," + attributes[counter_a] + ": ")
                f.write(str(float(pair_score)) + "\n")
            counter_a = counter_a +1
        counter_t = counter_t+1
    return pair_list

def generate_weat_results(filename_input, targets1, targets2, attributes1, attributes2):
    embedding_models = ["glove.6B.50d.txt", "glove.6B.100d.txt", "glove.6B.200d.txt", "glove.6B.300d.txt", "numberbatch-en-17.06.txt"]
    weat = WordEmbeddingAssociationTest()
    results_filename = "output/" + filename_input
    pair_results_filename = "output/pairs_" + filename_input
    
    # creating directories with file output - https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    with open(results_filename, "w") as f: # overwrite results file
        pass
    with open(pair_results_filename, "w") as f:
        pass

    for i in embedding_models:
        no_header = True # reset flags to suit majority of the embedding models
        is_binary = False
        print("Loading embeddings " + i)
        path = ("embeddings/" + i)
    
        if i ==("numberbatch-en-17.06.txt"):
            no_header = False
        if i == ("GoogleNews-vectors-negative300.bin"):
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
        X = to_tensor(temp_targets1, embeddings)      # target group 1
        Y = to_tensor(temp_targets2, embeddings)      # target group 2
        A = to_tensor(temp_attributes1, embeddings)   # attribute group 1
        B = to_tensor(temp_attributes2, embeddings)   # attribute group 2
        
        score = float(weat(X, Y, A, B))
        # Get cosine similarities for all combinations of target-attribute pairs
        with open(pair_results_filename, "a") as f:
            f.write("\nEmbedding model: " + i + "\n")
        pair_list = []
        generate_pair_values(X, temp_targets1, A, temp_attributes1, pair_results_filename, pair_list) # target group 1 - attribute group 1
        generate_pair_values(X, temp_targets1, B, temp_attributes2, pair_results_filename, pair_list) # target group 1 - attribute group 1
        generate_pair_values(Y, temp_targets2, A, temp_attributes1, pair_results_filename, pair_list) # target group 1 - attribute group 1
        generate_pair_values(Y, temp_targets1, B, temp_attributes1, pair_results_filename, pair_list) # target group 1 - attribute group 1
        t_stat, p_value = stats.ttest_1samp(pair_list, 0)
        
        with open(results_filename, "a") as f:
            f.write(i + "\n\tWEAT: "  + str(score) + "\n")
            f.write("\tT-Statistic: " + str(t_stat) + "\n")
            f.write("\tP-Value: " + str(p_value) + "\n\n")