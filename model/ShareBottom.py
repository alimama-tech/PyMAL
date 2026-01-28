from .BASE import BASE
import torch.nn.functional as F


class ShareBottom(BASE):
    def __init__(self, vocabs, num_feat_user, num_feat_ad, num_feat_seq, hidden_dim, main_view, aux_views):
        super().__init__(vocabs, num_feat_user, num_feat_ad, num_feat_seq, hidden_dim, main_view, aux_views)

    def forward(self, user_indices, ad_indices, buy_mm_scores, buy_seq_indices):
        fused = self.pre_forward(user_indices, ad_indices, buy_mm_scores, buy_seq_indices)
        fc1 = self.EXPERT['expert'](fused)

        probs = {}
        for view in self.aux_views + [self.main_view]:
            fc2 = self.AC(self.FC[view + 'fc2'](fc1))
            fc3 = self.AC(self.FC[view + 'fc3'](fc2))
            fc4 = self.AC(self.FC[view + 'fc4'](fc3))
            fc5 = self.FC[view + 'fc5'](fc4)

            probs[view] = F.softmax(fc5, dim=-1)

        return probs
