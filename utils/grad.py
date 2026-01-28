import torch
from .data import EPSILON


def get_pcgrad(main_loss, aux_loss_list, parameters):
    g_main = torch.autograd.grad(main_loss, parameters, retain_graph=True, allow_unused=True)

    total_g_aux = [None for _ in parameters]
    for aux_loss in aux_loss_list:
        g_aux = torch.autograd.grad(aux_loss, parameters, retain_graph=True, allow_unused=True)
        
        g_aux_corrected = []
        for gm, ga in zip(g_main, g_aux):
            if gm is None or ga is None:
                g_aux_corrected.append(ga)
                continue
            dot = torch.dot(gm.view(-1), ga.view(-1))
            if dot < 0:
                norm_sq = torch.dot(gm.view(-1), gm.view(-1)) + EPSILON
                ga = ga - (dot / norm_sq) * gm
            g_aux_corrected.append(ga)

        for i, ga in enumerate(g_aux_corrected):
            if ga is not None:
                if total_g_aux[i] is None:
                    total_g_aux[i] = ga.clone()
                else:
                    total_g_aux[i] += ga

    g_final = []
    for gm, ga in zip(g_main, total_g_aux):
        if gm is not None and ga is not None:
            g_final.append(gm + ga)
        elif gm is not None:
            g_final.append(gm)
        else:
            g_final.append(ga)
    return g_final


def cosine_similarity_grad(grads1, grads2):
    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for g1, g2 in zip(grads1, grads2):
        if g1 is None or g2 is None:
            continue
        dot_product += torch.sum(g1 * g2)
        norm1 += torch.sum(g1 ** 2)
        norm2 += torch.sum(g2 ** 2)
    norm1 = torch.sqrt(norm1)
    norm2 = torch.sqrt(norm2)
    return dot_product / (norm1 * norm2)


def get_gcsgrad(main_loss, aux_loss_list, parameters):
    g_main = torch.autograd.grad(main_loss, parameters, retain_graph=True, allow_unused=True)

    total_grad = []
    for gm in g_main:
        total_grad.append(gm.clone() if gm is not None else None)

    for aux_loss in aux_loss_list:
        g_aux = torch.autograd.grad(aux_loss, parameters, retain_graph=True, allow_unused=True)
        cs = torch.sigmoid(cosine_similarity_grad(g_main, g_aux))
        assert(cs>=0 and cs<=1)
        for i, ga in enumerate(g_aux):
            if ga is not None:
                if total_grad[i] is None:
                    total_grad[i] = ga.clone()
                else:
                    total_grad[i] += ga * cs
    return total_grad
