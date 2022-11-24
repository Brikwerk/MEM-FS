
import torch


def top_k_labels(support, query, k, labels):
    m = support.shape[0]
    n = query.shape[0]
    logits = -((query.unsqueeze(1).expand(n, m, -1) -
                support.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    topn_labels = labels[torch.topk(logits, k=k, dim=1).indices]
    pred_labels = topn_labels.mode(1).values
    return pred_labels


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class NullLayer(torch.nn.Module):
    def forward(self, x):
        return x


def barlow_loss(a, b, lam=5e-3):
    a_norm = (a - a.mean(0)) / a.std(0)
    b_norm = (b - b.mean(0)) / b.std(0)

    N, D = a.shape

    corr = torch.mm(a_norm.T, b_norm) / N

    corr_diff = (corr - torch.eye(D, device=a.device)).pow(2)

    mask = ~torch.eye(a.shape[-1], dtype=bool).to(a.device)
    corr_diff[mask] = corr_diff[mask].mul_(lam)

    return corr_diff.sum()