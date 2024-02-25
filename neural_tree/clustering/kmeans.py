import torch
from sklearn import cluster

__all__ = ["KMeans"]


def KMeans(
    documents_embeddings: torch.Tensor,
    n_clusters: int,
    max_iter: int,
    n_init: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, list]:
    """KMeans clustering."""
    kmeans: cluster.KMeans = cluster.KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_init=n_init,
        random_state=seed,
    ).fit(X=documents_embeddings)

    node_embeddings = torch.tensor(
        data=kmeans.cluster_centers_,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    return (
        node_embeddings,
        kmeans.labels_,
    )
