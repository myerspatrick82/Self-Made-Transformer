import pickle
from tokenizer import Tokenizer

with open("tokenizer_new.pkl", "rb") as file:
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
print(len(loaded_tokenizer.vocab))
# with open("testout.txt", "w") as file:
#     file.write(' '.join(loaded_tokenizer.vocab))
# print(loaded_tokenizer.vocab)