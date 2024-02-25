import os

__all__ = ["load_beir", "load_beir_train", "load_beir_test"]


def _make_pairs(queries: dict, qrels: dict) -> tuple[list, list]:
    """Make pairs of queries and documents for training."""
    test_queries, test_documents = [], []
    for query, (_, documents_queries) in zip(queries, qrels.items()):
        for document_id in documents_queries:
            test_queries.append(query)
            test_documents.append({"id": document_id})
    return test_queries, test_documents


def load_beir(dataset_name: str, split: str) -> tuple:
    """Load BEIR dataset."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    path = f"./beir_datasets/{dataset_name}"
    if not os.path.isdir(s=path):
        path = util.download_and_unzip(
            url=f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
            out_dir="./beir_datasets/",
        )

    documents, queries, qrels = GenericDataLoader(data_folder=path).load(split=split)

    documents = [
        {
            "id": document_id,
            "title": document["title"],
            "text": document["text"],
        }
        for document_id, document in documents.items()
    ]

    return documents, queries, qrels


def load_beir_train(dataset_name: str) -> tuple[list, list, list]:
    """Load training dataset.

    Parameters
    ----------
    dataset_name
        Dataset name

    Examples
    --------
    >>> from neural_tree import datasets

    >>> documents, train_queries, train_documents = datasets.load_beir_train(
    ...     dataset_name="scifact",
    ... )

    >>> len(documents)
    5183

    >>> assert len(train_queries) == len(train_documents)

    """
    documents, queries, qrels = load_beir(dataset_name=dataset_name, split="train")

    train_queries, train_documents = _make_pairs(
        queries=list(queries.values()), qrels=qrels
    )

    return documents, train_queries, train_documents


def load_beir_test(dataset_name: str) -> tuple[list, list, dict]:
    """Load BEIR testing dataset.

    Parameters
    ----------
    dataset_name
        Dataset name.

    Examples
    --------
    >>> from neural_tree import datasets

    >>> documents, queries_ids, queries, qrels = datasets.load_beir_test(
    ...     dataset_name="scifact",
    ... )

    >>> len(documents)
    5183

    >>> assert len(queries_ids) == len(queries) == len(qrels)
    """
    documents, queries, qrels = load_beir(dataset_name=dataset_name, split="test")
    return documents, list(queries.keys()), list(queries.values()), qrels
