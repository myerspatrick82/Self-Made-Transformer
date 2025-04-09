from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

trainer = trainers.BpeTrainer(
    vocab_size=2000,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)
tokenizer.train([
    "output.txt"
], trainer=trainer)

tokenizer.save("byte-level-bpe.tokenizer.json", pretty=True)

tokenizer = Tokenizer.from_file("byte-level-bpe.tokenizer.json")

encoded = tokenizer.encode("I can feel the magic, can you?")
print(encoded.tokens)

