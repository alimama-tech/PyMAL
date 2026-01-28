import torch
from .BASE import BASE
import torch.nn.functional as F
from .net.module import ExpertNet


class MMoE(BASE):
    def __init__(self, vocabs, num_feat_user, num_feat_ad, num_feat_seq, hidden_dim, main_view, aux_views, num_expert=2):
        super().__init__(vocabs, num_feat_user, num_feat_ad, num_feat_seq, hidden_dim, main_view, aux_views)

        self.num_expert = num_expert

        for aux_view in self.aux_views + [self.main_view]:
            self.FC[aux_view + 'gate'] = torch.nn.Linear(self.fc1_in_hidden_dim, self.num_expert)
        for i in range(self.num_expert):
            self.EXPERT['expert' + str(i)] = ExpertNet(self.fc1_in_hidden_dim, self.fc1_out_hidden_dim)

    def forward(self, user_indices, ad_indices, buy_mm_scores, buy_seq_indices):
        fused = self.pre_forward(user_indices, ad_indices, buy_mm_scores, buy_seq_indices)

        fc1_list = []
        for i in range(self.num_expert):
            fused_expert = self.EXPERT['expert' + str(i)](fused)
            fc1_list.append(fused_expert)

        probs = {}
        for view in self.aux_views + [self.main_view]:
            gate_weights = self.FC[view + 'gate'](fused)
            gate_weights = F.softmax(gate_weights, dim=-1)
            gate_weights = torch.unsqueeze(gate_weights, axis=-1)
            fc1_feats = torch.stack(fc1_list, axis=1)
            fused_fc1 = torch.matmul(fc1_feats.transpose(-2, -1), gate_weights)
            fc1 = torch.squeeze(fused_fc1, axis=2)

            fc2 = self.AC(self.FC[view + 'fc2'](fc1))
            fc3 = self.AC(self.FC[view + 'fc3'](fc2))
            fc4 = self.AC(self.FC[view + 'fc4'](fc3))
            fc5 = self.FC[view + 'fc5'](fc4)

            probs[view] = F.softmax(fc5, dim=-1)

        return probs
