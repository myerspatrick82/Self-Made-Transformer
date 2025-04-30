import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramNegSampling, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center_words, pos_context_words, neg_context_words):
        center_embeds = self.in_embed(center_words)
        pos_embeds = self.out_embed(pos_context_words)
        neg_embeds = self.out_embed(neg_context_words)

        pos_score = torch.sum(center_embeds * pos_embeds, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(neg_embeds.neg(), center_embeds.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(neg_score).sum(1)

        loss = -(pos_loss + neg_loss).mean()
        return loss