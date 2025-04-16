import pickle
from tokenizer import Tokenizer

with open("tokenizer2.pkl", "rb") as file:
    loaded_tokenizer = pickle.load(file)

def tokenize(text):
    return loaded_tokenizer.tokenize(text)

def encode(tokenized):
    return loaded_tokenizer.encode(tokenized)

def decode(encoded):
    return loaded_tokenizer.decode(encoded)

print(encode(tokenize("this is checkpointing")))
print(tokenize("this is checkpointing"))
print(encode(tokenize("this was violently cool")))
print(decode(encode(tokenize("this was violently cool"))))
print(tokenize("forth"))
print(decode(encode(tokenize("why am i waiting for this"))))