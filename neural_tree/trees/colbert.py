from neural_cherche import models

from ..clustering import get_mapping_nodes_documents
from ..scoring import ColBERT as scoring_ColBERT
from .tfidf import TfIdf
from .tree import Tree

__all__ = ["ColBERT"]


class ColBERT(Tree):
    def __init__(
        self,
        key: str,
        on: str | list[str],
        model: models.ColBERT,
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
        elif graph is None:
            tfidf = TfIdf(
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

            graph = tfidf.to_json()

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
