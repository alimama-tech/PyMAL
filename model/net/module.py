import torch
from torch import nn
from utils.data import USER_COLS, AD_COLS, SEQ_COLS, MAP_COLS


class SENET(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4, bias=False), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim // 4, self.hidden_dim, bias=False), 
            nn.Sigmoid()
        )

    def forward(self, x):
        assert self.input_dim % self.hidden_dim == 0
        x = x.view(-1, self.input_dim // self.hidden_dim, self.hidden_dim)
        y = x.mean(dim=1)
        y = self.fc(y)
        y = y.unsqueeze(1)
        output = x * y.expand_as(x)
        return output.view(-1, self.input_dim)


class ExpertNet(nn.Module):
    def __init__(self, in_hidden, out_hidden):
        super().__init__()
        self.senet = SENET(in_hidden)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_hidden, out_hidden)
    
    def forward(self, x):
        x = self.senet(x)
        x = self.act(self.fc1(x))
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        attn_out, _ = self.cross_attn(query=query, key=key, value=value, key_padding_mask=None)
        return attn_out


class DeepSetLevelEmbedding(nn.Module):
    def __init__(self, eps=1/16, dim=1):
        super().__init__()
        self.eps = eps
        self.bias = int(1 / eps)
        self.num_bins = 2 * self.bias
        self.dim = dim
        self.bin_embs = nn.Parameter(torch.randn(self.num_bins, self.dim) * 0.01)

    def forward(self, cosine):
        cosine = torch.clamp(cosine, -0.999, 0.999)
        ids = torch.floor(cosine / self.eps).long() + self.bias
        range_bins = torch.arange(0, self.num_bins, device=cosine.device).view(1, self.num_bins, 1)
        weight = (range_bins == ids.unsqueeze(1)).float()
        log_counts = torch.log2(weight.sum(dim=2, keepdim=True) + 1.0)
        bin_embs = self.bin_embs.unsqueeze(0)
        weighted_bins = log_counts * bin_embs
        output = weighted_bins.view(cosine.size(0), -1)
        return output


class EmbeddingTable(nn.Module):
    def __init__(self, vocabs, hidden_dim):
        super().__init__()
        self.vocabs = vocabs
        self.embs_table = nn.ModuleDict({col : nn.Embedding(len(vocabs[col]), hidden_dim) for col in USER_COLS[1:] + AD_COLS})

    def forward(self, user_indices, ad_indices, buy_seq_indices):
        user_embs = []
        for i, col in enumerate(USER_COLS[1:]):
            idx_col = user_indices[:, i]
            emb = self.embs_table[col](idx_col)
            user_embs.append(emb)
        user_embs = torch.stack(user_embs, dim=1)

        ad_embs = []
        for i, col in enumerate(AD_COLS):
            idx_col = ad_indices[:, i]
            emb = self.embs_table[col](idx_col)
            ad_embs.append(emb)
        ad_embs = torch.stack(ad_embs, dim=1)

        buy_seq_embs = []
        for i, col in enumerate(SEQ_COLS):
            idx_col = buy_seq_indices[:, i, :]
            emb = self.embs_table[MAP_COLS[col]](idx_col)
            buy_seq_embs.append(emb)
        buy_seq_embs = torch.stack(buy_seq_embs, dim=1)

        return user_embs, ad_embs, buy_seq_embs
