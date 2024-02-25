import collections

import torch
from joblib import Parallel, delayed

from ..clustering import KMeans, average
from ..leafs import Leaf
from ..scoring import SentenceTransformer, TfIdf

__all__ = ["Node"]


class Node(torch.nn.Module):
    """Node of the tree."""

    def __init__(
        self,
        level: int,
        key: str,
        documents_embeddings: torch.Tensor,
        documents: list,
        leaf_balance_factor: int,
        branch_balance_factor: int,
        device: str,
        node_name: int | str,
        scoring: SentenceTransformer | TfIdf,
        seed: int,
        max_iter: int,
        n_init: int,
        parent: int,
        n_jobs: int,
        create_retrievers: bool,
        graph: dict | None,
    ) -> None:
        super(Node, self).__init__()
        self.level = level
        self.leaf_balance_factor = leaf_balance_factor
        self.branch_balance_factor = branch_balance_factor
        self.device = device
        self.seed = seed
        self.parent = parent
        self.node_name = node_name

        if graph is not None:
            self.nodes_embeddings, labels, documents, documents_embeddings = average(
                key=key,
                documents=documents,
                documents_embeddings=documents_embeddings,
                graph=graph,
                scoring=scoring,
                device=self.device,
            )
        else:
            self.nodes_embeddings, labels = KMeans(
                documents_embeddings=documents_embeddings,
                n_clusters=self.branch_balance_factor,
                max_iter=max_iter,
                n_init=n_init,
                seed=self.seed,
                device=self.device,
            )

        clusters = collections.defaultdict(list)
        for document, embedding, group in zip(documents, documents_embeddings, labels):
            clusters[group].append((document, embedding))

        if n_jobs == 1:
            self.childs = [
                self.create_child(
                    level=self.level + 1,
                    node_name=f"{self.node_name}{group}" if graph is None else group,
                    key=key,
                    documents=[document for document, _ in clusters[group]],
                    documents_embeddings=[
                        embedding for _, embedding in clusters[group]
                    ],
                    scoring=scoring,
                    max_iter=max_iter,
                    n_init=n_init,
                    create_retrievers=create_retrievers,
                    graph=graph[group] if graph is not None else None,
                    n_jobs=n_jobs,
                    seed=self.seed,
                )
                for group in sorted(
                    clusters, key=lambda key: len(clusters[key]), reverse=True
                )
            ]
        else:
            self.childs = Parallel(n_jobs=n_jobs)(
                delayed(function=self.create_child)(
                    level=self.level + 1,
                    node_name=f"{self.node_name}{group}" if graph is None else group,
                    key=key,
                    documents=[document for document, _ in clusters[group]],
                    documents_embeddings=[
                        embedding for _, embedding in clusters[group]
                    ],
                    scoring=scoring,
                    max_iter=max_iter,
                    n_init=n_init,
                    create_retrievers=create_retrievers,
                    graph=graph[group] if graph is not None else None,
                    n_jobs=n_jobs,
                    seed=self.seed,
                )
                for group in sorted(
                    clusters, key=lambda key: len(clusters[key]), reverse=True
                )
            )

    def create_child(
        self,
        level: int,
        node_name: str,
        key: str,
        documents: list,
        documents_embeddings: list,
        scoring: SentenceTransformer | TfIdf,
        max_iter: int,
        n_init: int,
        create_retrievers: bool,
        graph: dict | list | None,
        n_jobs: int,
        seed: int,
    ) -> None:
        """Create a child."""
        child = Leaf if len(documents) <= self.leaf_balance_factor else Node
        if graph is not None and isinstance(graph, list):
            child = Leaf

        child = child(
            level=level,
            node_name=node_name,
            key=key,
            scoring=scoring,
            documents=documents,
            documents_embeddings=scoring.stack(embeddings=documents_embeddings),
            leaf_balance_factor=self.leaf_balance_factor,
            branch_balance_factor=self.branch_balance_factor,
            device=self.device,
            seed=seed,
            max_iter=max_iter,
            n_init=n_init,
            parent=self.node_name,
            create_retrievers=create_retrievers,
            graph=graph,
            n_jobs=n_jobs,
        )
        return child

    def __str__(self) -> str:
        """String representation of a"""
        sep = "\t"
        return f"{self.level * sep} node {self.node_name}"

    def nodes_scores(
        self,
        scoring: SentenceTransformer | TfIdf,
        queries_embeddings: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Return the scores of the embeddings."""
        return scoring.nodes_scores(
            queries_embeddings=queries_embeddings,
            nodes_embeddings=self.nodes_embeddings,
        )

    def get_childs_and_scores(
        self,
        queries: list,
        scores: torch.Tensor,
        tree_scores: collections.defaultdict,
        paths: list | None,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the childs and scores given matrix of scores."""
        if paths is None:
            scores = torch.stack(tensors=scores, dim=1)
            scores, childs = torch.topk(input=scores, k=min(k, scores.shape[1]), dim=1)
            return childs, scores

        # If paths is not None, we go through the choosen path.
        path = [query_path.pop(0) if query_path else leaf for leaf, query_path in paths]

        child_node_names = []
        for node_name in path:
            for index, child in enumerate(iterable=self.childs):
                if node_name == child.node_name:
                    child_node_names.append(index)
                    break

        childs = torch.tensor(
            data=child_node_names,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(dim=1)

        scores = torch.stack(
            tensors=[tree_scores[query][node] for query, node in zip(queries, path)],
            dim=0,
        ).unsqueeze(dim=1)

        return childs, scores

    def search(
        self,
        queries: list[str],
        queries_embeddings: torch.Tensor,
        scoring: SentenceTransformer | TfIdf,
        k: int,
        beam_search_depth: int,
        paths: dict[list] | None = None,
        tree_scores: collections.defaultdict | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, list]:
        """Search for the closest embedding.

        Parameters
        ----------
        queries:
            Queries to search for.
        embeddings:
            Embeddings to search for.
        tree_scores:
            Dictionnary of already computed scores in the tree.
        documents:
            Documents add to the leafs.
        k:
            node_name of closest embeddings to return.
        paths:
            Paths to explore.
        """
        # We go through the choosen path and we do not explore the tree if paths.
        if paths is not None:
            k = 1

        # Store childs scores:
        if tree_scores is None:
            tree_scores = collections.defaultdict(dict)

        scores = []
        for index, node in enumerate(iterable=self.childs):
            score = node.nodes_scores(
                scoring=scoring,
                queries_embeddings=queries_embeddings,
                node_embedding=self.nodes_embeddings[index],
            )

            scores.append(score)
            for query, query_score in zip(queries, score):
                tree_scores[query][node.node_name] = query_score

        childs, scores = self.get_childs_and_scores(
            queries=queries,
            scores=scores,
            tree_scores=tree_scores,
            paths=paths,
            k=k if self.level == beam_search_depth else 1,
        )

        # Aggregate embeddings by child.
        index_embeddings, index_queries, index_paths = (
            collections.defaultdict(list),
            collections.defaultdict(list),
            collections.defaultdict(list),
        )

        for index, query_childs in enumerate(iterable=childs):
            for child in query_childs.tolist():
                index_embeddings[child].append(queries_embeddings[index])
                index_queries[child].append(queries[index])

            if paths is not None:
                index_paths[child].append(paths[index])

        index_paths = dict(index_paths)

        for (child, embeddings), (_, queries_child) in zip(
            index_embeddings.items(),
            index_queries.items(),
        ):
            tree_scores = self.childs[child].search(
                queries=queries_child,
                queries_embeddings=torch.stack(tensors=embeddings, axis=0),
                scoring=scoring,
                tree_scores=tree_scores,
                paths=index_paths[child] if child in index_paths else None,
                k=k,
                beam_search_depth=beam_search_depth,
            )

        return tree_scores

    def to_json(self) -> dict:
        return {child.node_name: child.to_json() for child in self.childs}
