from collections import defaultdict
import pickle

class Tokenizer:
    
    def __init__(self, text, vocab_size=2000):
        self.text = text
        self.vocab_size = vocab_size
        self.vocab = None
        self.merges = None
        self.word_freqs = None
        self.pair_freqs = None
    
    def train(self):
        self.word_freqs = self._word_freq()
        self.vocab = self._create_base_vocabulary()
        splits = {word: [c for c in word] for word in self.word_freqs.keys()}
        self.pair_freqs = self._compute_pair_freqs(splits)
        self.merges = self._merge(splits)

    def alphabet(self):
        pass

    def tokenize(self, text):
        pre_tokenized_text = text.split()
        splits = [[l for l in word] for word in pre_tokenized_text]

        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split

        return sum(splits, [])

    def encode(self, tokenized):
        encoded = []
        for token in tokenized:
            for i, v in enumerate(self.vocab):
                if token == v:
                    encoded.append(i)
        return encoded

    def decode(self, encoded):
        decoded = [self.vocab[x] for x in encoded]
        return decoded

    def _create_base_vocabulary(self):
        vocab = []
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in vocab:
                    vocab.append(letter)
        vocab.sort()
        vocab = ["<|endoftext|>"] + vocab
        return vocab

    def _merge_pair(self, a, b, splits):  # splits is each word and its split IE {word: ["w","o","r","d"]}
        for word in self.word_freqs:  # loops through all the words and their freq
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

    def _merge(self, splits):
        import time
        start = time.time()
        self.merges = {}
        vocab_size = 2000
        iterations = 0
        print("in merge")
        has_200 = False
        while len(self.vocab) < vocab_size:  # finds best pair, merges it, and adds it
            if iterations % 50 == 0 and iterations > 0:
                print("50 iterations passed, checkpointing")
                with open("tokenizer2.pkl", "wb") as file:
                    pickle.dump(self, file)
                has_200 = True
            if has_200:
                print(iterations)
            self.pair_freqs = self._compute_pair_freqs(splits)
            best_pair = ""
            max_freq = None
            for pair in self.pair_freqs:
                freq = self.pair_freqs[pair]
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            splits = self._merge_pair(*best_pair, splits)  # merge the pair together in splits
            self.merges[best_pair] = best_pair[0] + best_pair[1]  
            self.vocab.append(best_pair[0] + best_pair[1])
            iterations += 1
        end = time.time()
        print(end-start, "time")
        return self.merges
    
    def _compute_pair_freqs(self, splits):  # find pair frequencies of ALL words, I.E. ("word", 2)
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def _word_freq(self):
        """
        Creates a defaultdict with each word and its frequency

        Args:
            corpus: specified text bank

        Returns:
            {word: freq} counts
        """
        word_freqs = defaultdict(int)
        for word in self.text.split():
            word_freqs[word] += 1
        return word_freqs
    

def main():
    # test corpus
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

    # tokenizer = AutoTokenizer.from_pretrained("gpt2")  # pre-processing for splitting tokens

    # text = open("output.txt", "r", encoding="utf-8").read()
    print("opening file")
    # text = open("output.txt", "r", encoding='utf-8').read()
    text = open("out.txt", "r").read()
    text.encode("utf-8", errors="ignore").decode("utf-8")
    # some tests
    print("initializing tokenizer class")
    tokenizer = Tokenizer(text=text)
    print("training")
    tokenizer.train()

    with open("tokenizer.pkl", "wb") as file:
        pickle.dump(tokenizer, file)

    print("Tokenizer saved to tokenizer.pkl")

    tokenized = tokenizer.tokenize("What a cool tokenizer")
    encoded = tokenizer.encode(tokenized)
    decoded = tokenizer.decode(encoded)
    print(tokenized)
    print(encoded)
    print(decoded)

if __name__ == "__main__":
    main()