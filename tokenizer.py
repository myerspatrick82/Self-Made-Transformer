import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from collections import defaultdict

corpus = [
    "Natural language processing is a fascinating field.",
    "Tokenizers break text into smaller units called tokens.",
    "BPE helps create subword units by merging frequent pairs.",
    "The transformer architecture is central to modern NLP.",
    "Machine learning models need lots of data and compute.",
    "Training a language model takes time and resources.",
    "This example shows how tokenization can vary.",
    "Let's explore how different algorithms tokenize the same text.",
]

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # pre-processing for splitting tokens

# create word frequencies---------------------
##############################################

text = open("output.txt", "r", encoding="utf-8").read()
text.encode("utf-8", errors="ignore").decode("utf-8")

def word_freq(corpus):
    """
    Creates a defaultdict with each word and its frequency

    Args:
        corpus: specified text bank

    Returns:
        {word: freq} counts
    """
    word_freqs = defaultdict(int)
    # for text in corpus: 
    #     # words_with_offset = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    #     # new_words = [word for word, offset in words_with_offset]
    #     # for word in new_words:
    #         # word_freqs[word] += 1
    #     words = text.split()
    #     for word in words:
    #         word_freqs[word] += 1
    for word in corpus.split():
        word_freqs[word] += 1
    return word_freqs

# word_freqs = word_freq(corpus)
word_freqs = word_freq(text)
print(word_freqs)

# create vocab section------------------------
##############################################

def create_base_vocabulary(word_freqs):
    vocab = []
    for word in word_freqs.keys():
        for letter in word:
            if letter not in vocab:
                vocab.append(letter)
    vocab.sort()
    vocab = ["<|endoftext|>"] + vocab
    return vocab

vocab = create_base_vocabulary(word_freqs)

splits = {word: [c for c in word] for word in word_freqs.keys()}  # split each word into individual characters (using a dict)
# print(splits, "before")
def compute_pair_freqs(splits):  # find pair frequencies of ALL words, I.E. ("word", 2)
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

pair_freqs = compute_pair_freqs(splits)

# merging pairs-------------------------------
##############################################

def merge_pair(a, b, splits):  # splits is each word and its split IE {word: ["w","o","r","d"]}
    for word in word_freqs:  # loops through all the words and their freq
        split = splits[word]
        if len(split) == 1:  # check if theres only one split ie one character and go to next word if one
            continue

        i = 0
        while i < len(split) - 1:  # else we run through each character in the word
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

def merge(splits, vocab):
    merges = {}
    vocab_size = 2000
    while len(vocab) < vocab_size:  # finds best pair, merges it, and adds it
        pair_freqs = compute_pair_freqs(splits)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():  # loop through all pair, freq
            if max_freq is None or max_freq < freq: 
                best_pair = pair  # found best
                max_freq = freq  # set to freq
        splits = merge_pair(*best_pair, splits)  # merge the pair together in splits
        merges[best_pair] = best_pair[0] + best_pair[1]  
        vocab.append(best_pair[0] + best_pair[1])
    return merges
merges = merge(splits, vocab)

def tokenize(text):
    # pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    # pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    pre_tokenized_text = text.split()
    # pre_tokenized_text = [word for word in text]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])

print(tokenize("What a cool tokenizer"))
tokenized = tokenize("What a cool tokenizer")
encoded = []
for token in tokenized:
    for i, v in enumerate(vocab):
        if token == v:
            encoded.append(i)
print(encoded)
decoded = [vocab[x] for x in encoded]
print(decoded)
print(vocab)
print(merges)

# TODO:

class Tokenizer:
    
    def __init__(self, text):
        pass
    
    def train():
        pass

    def alphabet():
        pass

