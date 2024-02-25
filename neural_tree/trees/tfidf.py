from sklearn.feature_extraction.text import TfidfVectorizer

from ..clustering import get_mapping_nodes_documents
from ..scoring import TfIdf as scoring_TfIdf
from .tree import Tree

__all__ = ["TfIdf"]


class TfIdf(Tree):
    """Tree with tfidf scoring function.

    Parameters
    ----------
    key
        Key to identify the documents.
    on
        List of columns to use for the retrieval.
    tfidf_nodes
        TfidfVectorizer for the nodes.
    tfidf_documents
        TfidfVectorizer for the documents.
    documents
        List of documents to index.
    graph
        Existing graph to initialize the tree.
    leaf_balance_factor
        Balance factor for the leafs. Once there is less than `leaf_balance_factor`
        documents in a node, the node becomes a leaf.
    branch_balance_factor
        Balance factor for the branches. The number of children of a node is limited to
        `branch_balance_factor`.
    device
        Device to use for the retrieval.
    n_jobs
        Number of jobs to use when creating the tree. If -1, all CPUs are used.
    max_iter
        Maximum number of iterations to perform with Kmeans algorithm when creating the
        tree.
    n_init
        Number of time the KMeans algorithm will be run with different centroid seeds.
    create_retrievers
        Whether to create the retrievers or not. If False, the tree is only created and
        the __call__ method will only output relevant leafs and scores rather than
        ranked documents.
    tqdm_bar
        Whether to show the tqdm bar when creating the tree.
    seed
        Random seed.

    Examples
    --------
    >>> from neural_tree import trees
    >>> from pprint import pprint

    >>> documents = [
    ...     {"id": 0, "text": "Paris is the capital of France."},
    ...     {"id": 1, "text": "Berlin is the capital of Germany."},
    ...     {"id": 2, "text": "Paris and Berlin are European cities."},
    ...     {"id": 3, "text": "Paris and Berlin are beautiful cities."},
    ... ]

    >>> tree = trees.TfIdf(
    ...    key="id",
    ...    on="text",
    ...    documents=documents,
    ...    leaf_balance_factor=2,
    ...    branch_balance_factor=2,
    ... )

    >>> tree = tree.add(documents=documents)

    >>> print(tree)
    node 1
        leaf 10
        leaf 11

    >>> tree.leafs_to_documents
    {'10': [0, 1], '11': [2, 3]}

    >>> candidates = tree(
    ...    queries=["Paris is the capital of France.", "Paris and Berlin are European cities."],
    ...    k_leafs=2,
    ...    k=2,
    ... )

    >>> candidates["scores"]
    array([[0.81927449, 0.10763316],
           [0.8641156 , 0.10763316]])

    >>> candidates["leafs"]
    array([['10', '11'],
           ['11', '10']], dtype='<U2')

    >>> pprint(candidates["tree_scores"])
    [{'10': tensor(0.8193), '11': tensor(0.1076)},
     {'10': tensor(0.1076), '11': tensor(0.8641)}]

    >>> pprint(candidates["documents"])
    [[{'id': 0, 'leaf': '10', 'similarity': 0.9999999999999978},
      {'id': 1, 'leaf': '10', 'similarity': 0.39941742405759667}],
     [{'id': 2, 'leaf': '11', 'similarity': 0.9999999999999978},
      {'id': 3, 'leaf': '11', 'similarity': 0.5385719658738707}]]

    """

    def __init__(
        self,
        key: str,
        on: str | list[str],
        documents: list[dict] | None = None,
        graph: dict | None = None,
        leaf_balance_factor: int = 100,
        branch_balance_factor: int = 5,
        tfidf_nodes: TfidfVectorizer | None = None,
        tfidf_documents: TfidfVectorizer | None = None,
        device: str = "cpu",
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
            scoring=scoring_TfIdf(
                key=key,
                on=on,
                documents=documents,
                tfidf_nodes=tfidf_nodes,
                tfidf_documents=tfidf_documents,
                device=device,
            ),
            leaf_balance_factor=leaf_balance_factor,
            branch_balance_factor=branch_balance_factor,
            device=device,
            n_jobs=n_jobs,
            create_retrievers=create_retrievers,
            max_iter=max_iter,
            n_init=n_init,
            seed=seed,
        )
