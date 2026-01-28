import torch
from torch import nn
from .BASE import BASE
import torch.nn.functional as F
from .net.module import ExpertNet


class MoAE(BASE):
    def __init__(self, vocabs, num_feat_user, num_feat_ad, num_feat_seq, hidden_dim, main_view, aux_views, num_public_expert=1, num_private_expert=1):
        super().__init__(vocabs, num_feat_user, num_feat_ad, num_feat_seq, hidden_dim, main_view, aux_views)

        self.num_public_expert = num_public_expert
        self.num_private_expert = num_private_expert
        self.num_expert = self.num_public_expert + self.num_private_expert

        for view in self.aux_views + [self.main_view]:
            self.FC[view + 'gate'] = nn.Linear((self.num_feat_seq + self.num_feat_ad + self.num_feat_user) * self.hidden_dim + self.deepset_dim, self.num_expert)
        for i in range(self.num_public_expert):
            self.EXPERT['expert' + str(i)] = ExpertNet((self.num_feat_seq + self.num_feat_ad + self.num_feat_user) * self.hidden_dim + self.deepset_dim, 256)
        for view in self.aux_views + [self.main_view]:
            for i in range(self.num_private_expert):
                self.EXPERT['expert' + view + str(i)] = ExpertNet((self.num_feat_seq + self.num_feat_ad + self.num_feat_user) * self.hidden_dim + self.deepset_dim, 256)
        
        self.FC['fusion2'] = nn.Linear(128 * self.num_view, 128)
        self.FC['fusion3'] = nn.Linear(64 * self.num_view , 64)
        self.FC['fusion4'] = nn.Linear(32 * self.num_view , 32)

    def forward(self, user_indices, ad_indices, buy_mm_scores, buy_seq_indices):
        fused = self.pre_forward(user_indices, ad_indices, buy_mm_scores, buy_seq_indices)

        fc1_public_list = []
        for i in range(self.num_public_expert):
            fused_expert = self.EXPERT['expert' + str(i)](fused)
            fc1_public_list.append(fused_expert)

        probs, aux_fc2, aux_fc3, aux_fc4 = {}, {}, {}, {}
        for aux_view in self.aux_views:
            fc1_private_list = []
            for i in range(self.num_private_expert):
                fused_expert = fused
                fused_expert = self.EXPERT['expert' + aux_view + str(i)](fused_expert)
                fc1_private_list.append(fused_expert)
            fc1_list = fc1_public_list + fc1_private_list
            fc1_feats = torch.stack(fc1_list, axis=1)
            gate_weights = self.FC[aux_view + 'gate'](fused)
            gate_weights = F.softmax(gate_weights, dim=-1)
            gate_weights = torch.unsqueeze(gate_weights, axis=-1)
            fused_fc1 = torch.matmul(fc1_feats.transpose(-2, -1), gate_weights)
            fc1 = torch.squeeze(fused_fc1, axis=2)

            fc2 = self.AC(self.FC[aux_view + 'fc2'](fc1))
            fc3 = self.AC(self.FC[aux_view + 'fc3'](fc2))
            fc4 = self.AC(self.FC[aux_view + 'fc4'](fc3))
            fc5 = self.FC[aux_view + 'fc5'](fc4)
            
            aux_fc2[aux_view], aux_fc3[aux_view], aux_fc4[aux_view] = fc2, fc3, fc4
            probs[aux_view] = F.softmax(fc5, dim=-1)

        view = self.main_view
        fc1_private_list = []
        for i in range(self.num_private_expert):
            fused_expert = fused
            fused_expert = self.EXPERT['expert' + view + str(i)](fused_expert)
            fc1_private_list.append(fused_expert)
        fc1_list = fc1_public_list + fc1_private_list
        fc1_feats = torch.stack(fc1_list, axis=1)
        gate_weights = self.FC[view + 'gate'](fused)
        gate_weights = F.softmax(gate_weights, dim=-1)
        gate_weights = torch.unsqueeze(gate_weights, axis=-1)
        fused_fc1 = torch.matmul(fc1_feats.transpose(-2, -1), gate_weights)
        fused_fc1 = torch.squeeze(fused_fc1, axis=2)

        aux_projs2 = [aux_fc2[aux_view] for aux_view in self.aux_views]
        fusion_input2 = torch.cat([self.FC[view + 'fc2'](fused_fc1)] + aux_projs2, dim=-1)
        fc2 = self.AC(self.FC['fusion2'](fusion_input2) + self.FC[view + 'fc2'](fused_fc1))

        aux_projs3 = [aux_fc3[aux_view] for aux_view in self.aux_views]
        fusion_input3 = torch.cat([self.FC[view + 'fc3'](fc2)] + aux_projs3, dim=-1)
        fc3 = self.AC(self.FC['fusion3'](fusion_input3) + self.FC[view + 'fc3'](fc2))

        aux_projs4 = [aux_fc4[aux_view] for aux_view in self.aux_views]
        fusion_input4 = torch.cat([self.FC[view + 'fc4'](fc3)] + aux_projs4, dim=-1)
        fc4 = self.AC(self.FC['fusion4'](fusion_input4) + self.FC[view + 'fc4'](fc3))

        fc5 = self.FC[view + 'fc5'](fc4)

        probs[self.main_view] = F.softmax(fc5, dim=-1)
        return probs
