import torch

from geoopt.manifolds import PoincareBallExact


def distortion_loss(
    embeddings: torch.Tensor, dist_targets: torch.Tensor, ball: PoincareBallExact
) -> torch.Tensor:
    embedding_dists = ball.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
    losses = (embedding_dists - dist_targets).abs() / dist_targets
    return losses.mean()
