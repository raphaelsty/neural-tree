from neural_cherche import models
from sentence_transformers import SentenceTransformer as SentenceTransformerModel

from ..clustering import get_mapping_nodes_documents
from ..scoring import ColBERT as scoring_ColBERT
from .sentence_transformer import SentenceTransformer
from .tfidf import TfIdf
from .tree import Tree

__all__ = ["ColBERT"]


class ColBERT(Tree):
    """ColBERT retriever.

    Parameters
    ----------
    key
        Key to identify the documents.
    on
        List of columns to use for the retrieval.
    model
        ColBERT model.
    sentence_transformer
        SentenceTransformer model in order to perform the hierarchical clustering. If
        None, the hierarchical clustering is performed with a TfIdf model.
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
    batch_size
        Batch size to use when creating the tree.
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

    """

    def __init__(
        self,
        key: str,
        on: str | list[str],
        model: models.ColBERT,
        sentence_transformer: SentenceTransformerModel | None = None,
        documents: list[dict] | None = None,
        graph: dict | None = None,
        leaf_balance_factor: int = 100,
        branch_balance_factor: int = 5,
        device: str = "cpu",
        n_jobs: int = -1,
        batch_size: int = 32,
        max_iter: int = 3000,
        n_init: int = 100,
        create_retrievers: bool = True,
        tqdm_bar: bool = True,
        seed: int = 42,
    ) -> None:
        """Create a tree with the TfIdf scoring."""
        if graph is not None:
            documents = get_mapping_nodes_documents(graph=graph)
        elif graph is None and sentence_transformer is None:
            index = TfIdf(
                key=key,
                on=on,
                documents=documents,
                leaf_balance_factor=leaf_balance_factor,
                branch_balance_factor=branch_balance_factor,
                create_retrievers=False,
                n_jobs=n_jobs,
                max_iter=max_iter,
                n_init=n_init,
                seed=seed,
            )
            graph = index.to_json()
        else:
            index = SentenceTransformer(
                key=key,
                on=on,
                documents=documents,
                model=sentence_transformer,
                leaf_balance_factor=leaf_balance_factor,
                branch_balance_factor=branch_balance_factor,
                n_jobs=n_jobs,
                create_retrievers=False,
                max_iter=max_iter,
                n_init=n_init,
                batch_size=batch_size,
                seed=seed,
            )
            graph = index.to_json()

        scoring = scoring_ColBERT(
            key=key,
            on=on,
            documents=documents,
            model=model,
            device=device,
        )

        # We computes embeddings here because we need documents contents.
        documents_embeddings = scoring.transform_documents(
            documents=documents,
            model=model,
            device=device,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
        )

        documents_embeddings = {
            document[key]: embedding
            for document, embedding in zip(documents, documents_embeddings)
        }

        super().__init__(
            key=key,
            graph=graph,
            documents=documents,
            scoring=scoring,
            documents_embeddings=documents_embeddings,
            leaf_balance_factor=leaf_balance_factor,
            branch_balance_factor=branch_balance_factor,
            device=device,
            n_jobs=1,
            create_retrievers=create_retrievers,
            max_iter=max_iter,
            n_init=n_init,
            seed=seed,
        )
