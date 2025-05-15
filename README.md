# Self-Made-Transformer
 
I wanted to better learn the Transformer and generative process of language models and the process that entails it, so this repo has some of the steps in the Transformer process. This includes...
+ Tokenization
+ Word Embeddings
+ Masked Multi-Headed Attention
+ Feed-Forward Network
+ Layer-norm 

## Tokenization
The Tokenizer was trained using Byte-Pair Encoding but was highly inefficient. The idea was to try to use my own Tokenizer to use when training the word embeddings but this idea eventually fell through as I realized my Tokenizer was not efficient enough and took a long time to encode the words compare to the NLTK tokenizer I ended up using. I ended up still training an initial tokenizer that took around ~12 hours to complete and had a vocabulary of 8000. 

## Word Embeddings

## Masked Multi-Headed Attention

## Feed-Forward Netowrk

## Layer-norm
