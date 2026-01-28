import torch
from torch import nn
import torch.nn.functional as F
from .net.module import ExpertNet, EmbeddingTable, CrossAttentionLayer, DeepSetLevelEmbedding


class BASE(nn.Module):
    def __init__(self, vocabs, num_feat_user, num_feat_ad, num_feat_seq, hidden_dim, main_view, aux_views):
        super().__init__()

        self.aux_views = aux_views
        self.main_view = main_view
        self.num_feat_user = num_feat_user
        self.num_feat_ad = num_feat_ad
        self.num_feat_seq = num_feat_seq
        self.hidden_dim = hidden_dim
        self.num_view = len(aux_views) + 1
        
        self.EmbeddingTable = EmbeddingTable(vocabs, hidden_dim)

        self.AC = nn.ReLU()

        self.FC = nn.ModuleDict()
        for view in self.aux_views + [self.main_view]:
            self.FC[view + 'fc2'] = nn.Linear(256, 128)
            self.FC[view + 'fc3'] = nn.Linear(128, 64)
            self.FC[view + 'fc4'] = nn.Linear(64 , 32)
            self.FC[view + 'fc5'] = nn.Linear(32 , 16 if view == 'cartesian' else 2)
        
        self.CrossAtten = nn.ModuleDict()
        for i in range(self.num_feat_seq):
            self.FC['query' + 'proj' + str(i)] = nn.Linear(self.num_feat_ad * self.hidden_dim, self.hidden_dim)
            self.CrossAtten[str(i)] = CrossAttentionLayer(self.hidden_dim)
        
        self.deepset_dim = 32
        self.DeepSetPic = DeepSetLevelEmbedding()
        self.DeepSetText = DeepSetLevelEmbedding()

        self.fc1_in_hidden_dim = (self.num_feat_seq + self.num_feat_ad + self.num_feat_user) * self.hidden_dim + self.deepset_dim
        self.fc1_out_hidden_dim = 256

        self.EXPERT = nn.ModuleDict()
        self.EXPERT['expert'] = ExpertNet(self.fc1_in_hidden_dim, self.fc1_out_hidden_dim)

        self.fused_norm = nn.LayerNorm(self.fc1_in_hidden_dim)

    def pre_forward(self, user_indices, ad_indices, buy_mm_scores, buy_seq_indices):
        user_embs, ad_embs, buy_seq_embs = self.EmbeddingTable(user_indices, ad_indices, buy_seq_indices)
        B = buy_seq_embs.shape[0]
        
        din = torch.concat([user_embs, ad_embs], dim=1)
        din = din.reshape(B, -1)
        
        atten_out = []
        for i in range(self.num_feat_seq):
            query = self.FC['query' + 'proj' + str(i)](ad_embs.reshape(B, 1, -1))
            key_value = buy_seq_embs[:, i]
            out = self.CrossAtten[str(i)](query, key_value, key_value).reshape(B, -1)
            atten_out.append(out)
        atten_out = torch.concat(atten_out, dim=1)

        deepset_out_pic = self.DeepSetPic(buy_mm_scores[:, 0, :])
        deepset_out_text = self.DeepSetText(buy_mm_scores[:, 1, :])
        deepset_out = deepset_out_pic + deepset_out_text

        fused = torch.concat([din, atten_out, deepset_out], dim=1)
        return self.fused_norm(fused)

    def forward(self, user_indices, ad_indices, buy_mm_scores, buy_seq_indices):
        fused = self.pre_forward(user_indices, ad_indices, buy_mm_scores, buy_seq_indices)
        fc1 = self.EXPERT['expert'](fused)

        view = self.main_view
        fc2 = self.AC(self.FC[view + 'fc2'](fc1))
        fc3 = self.AC(self.FC[view + 'fc3'](fc2))
        fc4 = self.AC(self.FC[view + 'fc4'](fc3))
        fc5 = self.FC[view + 'fc5'](fc4)

        probs = {}
        probs[self.main_view] = F.softmax(fc5, dim=-1)

        return probs


