import pickle
from tokenizer import Tokenizer

with open("tokenizer.pkl", "rb") as file:
    loaded_tokenizer = pickle.load(file)

def tokenize(text):
    return loaded_tokenizer.tokenize(text)

def encode(tokenized):
    return loaded_tokenizer.encode(tokenized)

print(encode(tokenize("this was violently cool")))