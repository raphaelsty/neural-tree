import numpy as np

__all__ = ["evaluate", "leafs_precision"]


def leafs_precision(
    key: str,
    documents: list,
    leafs: np.ndarray,
    documents_to_leaf: dict,
) -> float:
    """Calculate the precision of the leafs."""
    recall = 0
    for leafs_query, document in zip(leafs.tolist(), documents):
        for leaf_document in documents_to_leaf[document[key]]:
            if leafs_query[0] == leaf_document:
                recall += 1
                break
    return recall / len(leafs)


def evaluate(
    scores: list[list[dict]],
    qrels: dict,
    queries_ids: list[str],
    metrics: list = [],
    key: str = "id",
) -> dict[str, float]:
    """Evaluate candidates matchs.

    Parameters
    ----------
    matchs
        Matchs.
    qrels
        Qrels.
    queries
        index of queries of qrels.
    k
        Number of documents to retrieve.
    metrics
        Metrics to compute.

    Examples
    --------
    >>> from neural_cherche import models, retrieve, utils
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> model = models.Splade(
    ...     model_name_or_path="distilbert-base-uncased",
    ...     device="cpu",
    ... )

    >>> documents, queries_ids, queries, qrels = utils.load_beir(
    ...     "scifact",
    ...     split="test",
    ... )

    >>> documents = documents[:10]

    >>> retriever = retrieve.Splade(
    ...     key="id",
    ...     on=["title", "text"],
    ...     model=model
    ... )

    >>> documents_embeddings = retriever.encode_documents(
    ...     documents=documents,
    ...     batch_size=1,
    ... )

    >>> documents_embeddings = retriever.add(
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> queries_embeddings = retriever.encode_queries(
    ...     queries=queries,
    ...     batch_size=1,
    ... )

    >>> scores = retriever(
    ...     queries_embeddings=queries_embeddings,
    ...     k=30,
    ...     batch_size=1,
    ... )

    >>> utils.evaluate(
    ...     scores=scores,
    ...     qrels=qrels,
    ...     queries_ids=queries_ids,
    ...     metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"]
    ... )
    {'map': 0.0033333333333333335, 'ndcg@10': 0.0033333333333333335, 'ndcg@100': 0.0033333333333333335, 'recall@10': 0.0033333333333333335, 'recall@100': 0.0033333333333333335}

    """
    from ranx import Qrels, Run
    from ranx import evaluate as ranx_evaluate

    qrels = Qrels(qrels=qrels)

    run_dict = {
        id_query: {
            match[key]: 1 - (rank / len(query_matchs))
            for rank, match in enumerate(iterable=query_matchs)
        }
        for id_query, query_matchs in zip(queries_ids, scores)
    }

    run = Run(run=run_dict)

    if not metrics:
        metrics = ["ndcg@10"] + [f"hits@{k}" for k in [1, 2, 3, 4, 5, 10]]

    return ranx_evaluate(
        qrels=qrels,
        run=run,
        metrics=metrics,
        make_comparable=True,
    )
