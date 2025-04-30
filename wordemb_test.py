import numpy as np
import torch
import pickle
from numpy.linalg import norm

# -------- LOAD GLOVE --------
def load_glove(path):
    glove = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            word = tokens[0]
            vector = np.array(tokens[1:], dtype=np.float32)
            glove[word] = vector
    return glove

def normalize_embeddings(emb_dict):
    return {word: vec / norm(vec) for word, vec in emb_dict.items()}

# -------- CONVERT PYTORCH MATRIX TO DICT --------
def convert_custom_embeddings(matrix, idx_to_word):
    matrix = matrix / matrix.norm(dim=1, keepdim=True)
    return {idx_to_word[i]: matrix[i].numpy() for i in range(len(idx_to_word))}

# -------- GET TOP K SIMILAR WORDS --------
def get_top_k(word, embeddings, k):
    if word not in embeddings:
        raise ValueError(f"'{word}' not in embeddings.")
    vec = embeddings[word]
    similarities = {
        other: np.dot(vec, other_vec)
        for other, other_vec in embeddings.items() if other != word
    }
    top = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in top[:k]]

# -------- FIND SMALLEST K WITH OVERLAP --------
def find_min_k_with_overlap(word, emb1, emb2, max_k=100):
    for k in range(1, max_k + 1):
        top1 = set(get_top_k(word, emb1, k))
        top2 = set(get_top_k(word, emb2, k))
        if top1 & top2:
            return k, top1 & top2
    return None, set()

# -------- MAIN --------
def main():
    # Load GloVe
    print("Loading GloVe...")
    glove_raw = load_glove("glove.6B.50d.txt")
    glove = normalize_embeddings(glove_raw)

    # Load your custom checkpoint and vocab
    print("Loading custom embeddings and vocabulary...")
    checkpoint = torch.load("checkpoint_epoch_nltk5.pt", map_location="cpu")
    
    with open("idx_to_word.pkl", "rb") as f:
        idx_to_word = pickle.load(f)

    checkpoint = torch.load("checkpoint_epoch_nltk5.pt", map_location="cpu")

    from embedding import SkipGramNegSampling  # Replace with actual filename if needed

    vocab_size = checkpoint['vocab_size']
    embedding_dim = checkpoint['embedding_dim']
    model = SkipGramNegSampling(vocab_size, embedding_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extract embedding matrix
    embedding_matrix = model.in_embed.weight.data

    # Convert to word->vector dict
    custom_embeddings = convert_custom_embeddings(embedding_matrix, idx_to_word)

    # Run comparison
    words = ['king', 'queen', 'dog', 'cat', 'man', 'woman', 'girl', 'boy', 'computer', 'kingdom']
    for word in words:
        k, overlap = find_min_k_with_overlap(word, glove, custom_embeddings)
        print(f"First k with overlap: {k} for word: '{word}'")
        print(f"Overlapping words: {overlap}")

if __name__ == "__main__":
    main()
