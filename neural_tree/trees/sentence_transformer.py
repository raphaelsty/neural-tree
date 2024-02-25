from sentence_transformers import SentenceTransformer

from ..clustering import get_mapping_nodes_documents
from ..scoring import SentenceTransformer as scoring_SenrenceTransformer
from .tree import Tree

__all__ = ["SentenceTransformer"]


class SentenceTransformer(Tree):
    """Tree with Sentence Transformer scoring function.

    Examples
    --------
    >>> from neural_tree import trees
    >>> from sentence_transformers import SentenceTransformer
    >>> from pprint import pprint

    >>> documents = [
    ...     {"id": 0, "text": "Paris is the capital of France."},
    ...     {"id": 1, "text": "Berlin is the capital of Germany."},
    ...     {"id": 2, "text": "Paris and Berlin are European cities."},
    ...     {"id": 3, "text": "Paris and Berlin are beautiful cities."},
    ... ]

    >>> tree = trees.SentenceTransformer(
    ...    key="id",
    ...    on=["text"],
    ...    documents=documents,
    ...    model=SentenceTransformer("all-mpnet-base-v2"),
    ...    leaf_balance_factor=2,
    ...    branch_balance_factor=2,
    ...    device="cpu",
    ... )

    >>> tree = tree.add(documents=documents)

    >>> print(tree)
    node 1
        node 11
            leaf 110
            leaf 111
        leaf 10

    >>> tree.leafs_to_documents
    {'110': [2, 3, 1], '111': [0], '10': [1]}

    >>> candidates = tree(
    ...    queries=["Paris is the capital of France.", "Paris and Berlin are European cities."],
    ...    k_leafs=2,
    ...    k=1,
    ... )

    >>> candidates["scores"]
    array([[1.        , 0.76908004],
           [0.88792843, 0.82272887]])

    >>> candidates["leafs"]
    array([['111', '10'],
           ['110', '10']], dtype='<U3')

    >>> pprint(candidates["tree_scores"])
    [{'10': tensor(0.7691, device='mps:0'),
      '11': tensor(1., device='mps:0'),
      '110': tensor(0.6536, device='mps:0'),
      '111': tensor(1., device='mps:0')},
     {'10': tensor(0.8227, device='mps:0'),
      '11': tensor(0.8879, device='mps:0'),
      '110': tensor(0.8879, device='mps:0'),
      '111': tensor(0.6923, device='mps:0')}]

    >>> pprint(candidates["documents"])
    [[{'id': 0, 'leaf': '111', 'similarity': 1.0}],
     [{'id': 2, 'leaf': '110', 'similarity': 1.0}]]

    """

    def __init__(
        self,
        key: str,
        on: str | list[str],
        model: SentenceTransformer,
        documents: list[dict] | None = None,
        documents_embeddings: dict | None = None,
        graph: dict | None = None,
        leaf_balance_factor: int = 100,
        branch_balance_factor: int = 5,
        device: str = "cpu",
        faiss_device: str = "cpu",
        batch_size: int = 32,
        n_jobs: int = -1,
        max_iter: int = 3000,
        n_init: int = 100,
        create_retrievers: bool = True,
        seed: int = 42,
    ) -> None:
        """Create a tree with the TfIdf scoring."""
        if graph is not None:
            documents = get_mapping_nodes_documents(graph=graph)

        super().__init__(
            key=key,
            documents=documents,
            graph=graph,
            documents_embeddings=documents_embeddings,
            scoring=scoring_SenrenceTransformer(
                key=key,
                on=on,
                model=model,
                device=device,
                faiss_device=faiss_device,
            ),
            leaf_balance_factor=leaf_balance_factor,
            branch_balance_factor=branch_balance_factor,
            device=device,
            batch_size=batch_size,
            n_jobs=n_jobs,
            max_iter=max_iter,
            n_init=n_init,
            create_retrievers=create_retrievers,
            seed=seed,
        )
