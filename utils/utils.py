import torch
import numpy as np
from .data import EPSILON, USER_COLS, AD_COLS, SEQ_COLS


def set_seed(seed):
    import os, random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)


def print_config(args):
    print("=" * 90)
    print("Training Configuration")
    print("=" * 90)
    print(f"Model Name         : {args.model}")
    print(f"Learning Rate      : {args.lr}")
    print(f"Hidden Dimension   : {args.hidden_dim}")
    print(f"Auxiliary Loss λ   : {args.lamda}")
    print(f"PCGrad Enabled     : {args.pcgrad}")
    print(f"GCSGrad Enabled    : {args.gcsgrad}")
    print(f"Main View          : {args.main_view}")
    print(f"Auxiliary Views    : {args.aux_views}")
    print("=" * 90)


def get_model(name, vocabs, hidden_dim, main_view, aux_views):
    from model.BASE import BASE
    from model.ShareBottom import ShareBottom
    from model.MMoE import MMoE
    from model.PLE  import PLE
    from model.HoME import HoME
    from model.NATAL import NATAL
    from model.MoAE import MoAE
    model = {'BASE': BASE, 'NATAL': NATAL, 'PLE': PLE, 'MMoE': MMoE, 'ShareBottom': ShareBottom, "HoME": HoME, "MoAE": MoAE}
    return model[name](vocabs, len(USER_COLS) - 1, len(AD_COLS), len(SEQ_COLS), hidden_dim, main_view, aux_views)


def log_loss(probs, labels, main_view, aux_views):
    aux_view_loss = []
    for aux_view in aux_views:
        if aux_view not in probs:
            continue
        prob_aux = probs[aux_view]
        if aux_view == 'cartesian':
            labels_aux_view = torch.zeros_like(labels['last'], dtype=torch.int32)
            for i, v in enumerate(['last', 'first', 'linear', 'mta']):
                labels_aux_view += labels[v].to(torch.int32) * (2 ** i)
            labels_aux_view = labels_aux_view.flatten()
            aux_prob = prob_aux[torch.arange(prob_aux.shape[0]), labels_aux_view] + EPSILON
        else: 
            aux_prob = prob_aux[torch.arange(prob_aux.shape[0]), labels[aux_view]] + EPSILON

        aux_loss_ = -torch.mean(torch.log(aux_prob))
        aux_view_loss.append(aux_loss_)

    prob = probs[main_view][torch.arange(probs[main_view].shape[0]), labels[main_view]] + EPSILON
    loss = -torch.mean(torch.log(prob))
    return loss, aux_view_loss


def compute_auc_gauc(user_idxs, labels, probs):
    from sklearn.metrics import roc_auc_score
    sort_idx = np.argsort(user_idxs)
    sorted_users = user_idxs[sort_idx]
    sorted_labels = labels[sort_idx]
    sorted_probs = probs[sort_idx]

    unique_users, group_starts = np.unique(sorted_users, return_index=True)
    group_ends = np.append(group_starts[1:], len(sorted_users))

    user_aucs, user_weights = [], []
    for i in range(len(unique_users)):
        start, end = group_starts[i], group_ends[i]
        user_labels = sorted_labels[start:end]
        user_probs = sorted_probs[start:end]

        if len(np.unique(user_labels)) < 2:
            continue

        auc = roc_auc_score(user_labels, user_probs)
        pos_count = np.sum(user_labels == 1)
        neg_count = np.sum(user_labels == 0)
        weight = pos_count + neg_count
        
        user_aucs.append(auc)
        user_weights.append(weight)

    auc_global = roc_auc_score(labels, probs)
    n_valid = len(user_aucs)
    
    if n_valid == 0:
        gauc = 0.0
    else:
        gauc = np.average(user_aucs, weights=user_weights)

    return auc_global, gauc, n_valid
