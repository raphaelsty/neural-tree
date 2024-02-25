import collections
import copy
import random
from functools import lru_cache
from typing import Generator

import numpy as np
import torch
from scipy import sparse

from ..leafs import Leaf
from ..nodes import Node
from ..scoring import SentenceTransformer, TfIdf
from ..utils import sanity_check

__all__ = ["Tree"]


class Tree(torch.nn.Module):
    """Tree based index for information retrieval.

    Examples
    --------
    >>> from neural_tree import trees, scoring, clustering
    >>> from pprint import pprint

    >>> device = "cpu"

    >>> queries = [
    ...     "Paris is the capital of France.",
    ...     "Berlin",
    ...     "Berlin",
    ...     "Paris is the capital of France."
    ... ]

    >>> documents = [
    ...     {"id": 0, "text": "Paris is the capital of France."},
    ...     {"id": 1, "text": "Berlin is the capital of Germany."},
    ...     {"id": 2, "text": "Paris and Berlin are European cities."},
    ...     {"id": 3, "text": "Paris and Berlin are beautiful cities."},
    ... ]

    >>> tree = trees.Tree(
    ...    key="id",
    ...    documents=documents,
    ...    scoring=scoring.TfIdf(key="id", on=["text"], documents=documents),
    ...    leaf_balance_factor=1,
    ...    branch_balance_factor=2,
    ...    device=device,
    ...    n_jobs=1,
    ... )

    >>> print(tree)
    node 1
        node 10
            leaf 100
            leaf 101
        node 11
            leaf 110
            leaf 111

    >>> tree.documents_to_leafs
    {0: ['100'], 1: ['101'], 2: ['110'], 3: ['111']}

    >>> tree.leafs_to_documents
    {'100': [0], '101': [1], '110': [2], '111': [3]}

    >>> candidates = tree(
    ...    queries=queries,
    ...    k=2,
    ...    k_leafs=2,
    ... )

    >>> pprint(candidates["documents"])
    [[{'id': 0, 'leaf': '100', 'similarity': 0.9999999999999978},
      {'id': 1, 'leaf': '101', 'similarity': 0.39941742405759667}],
     [{'id': 3, 'leaf': '111', 'similarity': 0.3523828592933607},
      {'id': 2, 'leaf': '110', 'similarity': 0.348413283355546}],
     [{'id': 3, 'leaf': '111', 'similarity': 0.3523828592933607},
      {'id': 2, 'leaf': '110', 'similarity': 0.348413283355546}],
     [{'id': 0, 'leaf': '100', 'similarity': 0.9999999999999978},
      {'id': 1, 'leaf': '101', 'similarity': 0.39941742405759667}]]

    >>> pprint(candidates["tree_scores"])
    [{'10': tensor(1.0000),
      '100': tensor(1.0000),
      '101': tensor(0.6385),
      '11': tensor(0.1076)},
     {'10': tensor(0.3235),
      '11': tensor(0.3327),
      '110': tensor(0.3327),
      '111': tensor(0.3327)},
     {'10': tensor(0.3235),
      '11': tensor(0.3327),
      '110': tensor(0.3327),
      '111': tensor(0.3327)},
     {'10': tensor(1.0000),
      '100': tensor(1.0000),
      '101': tensor(0.6385),
      '11': tensor(0.1076)}]


    >>> candidates = tree(
    ...    queries=queries,
    ...    leafs=["110", "111", "111", "111"],
    ... )

    >>> pprint(candidates["documents"])
    [[{'id': 2, 'leaf': '110', 'similarity': 0.1036216271728989}],
     [{'id': 3, 'leaf': '111', 'similarity': 0.3523828592933607}],
     [{'id': 3, 'leaf': '111', 'similarity': 0.3523828592933607}],
     [{'id': 3, 'leaf': '111', 'similarity': 0.09981163726061484}]]

    >>> optimizer = torch.optim.AdamW(lr=3e-5, params=list(tree.parameters()))

    >>> loss = tree.loss(
    ...    queries=queries,
    ...    documents=documents,
    ... )

    >>> loss.backward()
    >>> optimizer.step()
    >>> assert loss.item() > 0

    >>> graph = tree.to_json()
    >>> pprint(graph)
    {1: {'10': {'100': [{'id': 0}], '101': [{'id': 1}]},
         '11': {'110': [{'id': 2}], '111': [{'id': 3}]}}}

    >>> graph = {'sport': {'football': {'bayern': [{'id': 2, 'text': 'bayern football team'}],
    ...             'psg': [{'id': 1, 'text': 'psg football team'}]},
    ...    'rugby': {'toulouse': [{'id': 3, 'text': 'toulouse rugby team'}],
    ...              'ville rose': [{'id': 3, 'text': 'toulouse rugby team'},
    ...                             {'id': 4, 'text': 'tfc football team'}]}}}

    >>> documents = clustering.get_mapping_nodes_documents(graph=graph)

    >>> tree = trees.Tree(
    ...    key="id",
    ...    documents=documents,
    ...    scoring=scoring.TfIdf(key="id", on=["text"], documents=documents),
    ...    leaf_balance_factor=1,
    ...    branch_balance_factor=2,
    ...    device=device,
    ...    graph=graph,
    ...    n_jobs=1,
    ... )

    >>> tree.documents_to_leafs
    {3: ['ville rose', 'toulouse'], 4: ['ville rose'], 2: ['bayern'], 1: ['psg']}

    >>> tree.leafs_to_documents
    {'ville rose': [3, 4], 'toulouse': [3], 'bayern': [2], 'psg': [1]}

    >>> print(tree)
    node sport
        node rugby
            leaf ville rose
            leaf toulouse
        node football
            leaf bayern
            leaf psg

    >>> candidates = tree(
    ...    queries=["psg", "toulouse"],
    ...    k=2,
    ...    k_leafs=2,
    ... )

    >>> pprint(candidates["documents"])
    [[{'id': 1, 'leaf': 'psg', 'similarity': 0.5255159378077358}],
     [{'id': 3, 'leaf': 'ville rose', 'similarity': 0.7865788511708137},
      {'id': 3, 'leaf': 'toulouse', 'similarity': 0.7865788511708137}]]

    References
    ----------
    [Li et al., 2023](https://arxiv.org/pdf/2206.02743.pdf)

    """

    def __init__(
        self,
        key: str,
        scoring: SentenceTransformer | TfIdf,
        documents: list,
        leaf_balance_factor: int,
        branch_balance_factor: int,
        device,
        seed: int,
        max_iter: int,
        n_init: int,
        n_jobs: int,
        batch_size: int = None,
        create_retrievers: bool = True,
        graph: dict | None = None,
        documents_embeddings: dict | None = None,
    ) -> None:
        super(Tree, self).__init__()
        self.key = key
        self.device = device
        self.seed = seed
        self.scoring = scoring
        self.create_retrievers = create_retrievers
        self.node_name = 1

        # Sanity check over input parameters
        sanity_check(
            branch_balance_factor=branch_balance_factor,
            leaf_balance_factor=leaf_balance_factor,
            graph=graph,
            documents=documents,
        )

        if graph is not None:
            for node_name in graph.keys():
                self.node_name = node_name
                break

        if documents_embeddings is None:
            documents_embeddings = self.scoring.transform_documents(
                documents=documents, batch_size=batch_size
            )
        else:
            documents_embeddings = self.scoring.stack(
                embeddings=[
                    documents_embeddings[document[self.key]] for document in documents
                ]
            )

        self.tree = Node(
            level=0,
            node_name=self.node_name,
            key=self.key,
            documents=documents,
            documents_embeddings=documents_embeddings,
            scoring=scoring,
            leaf_balance_factor=leaf_balance_factor,
            branch_balance_factor=branch_balance_factor,
            device=self.device,
            seed=self.seed,
            n_jobs=n_jobs,
            create_retrievers=create_retrievers,
            graph=graph[self.node_name] if graph is not None else None,
            parent=0,
            max_iter=max_iter,
            n_init=n_init,
        )

        self.documents_to_leafs, self.leafs_to_documents = self.get_documents_leafs()
        self.negative_samples = self.get_negative_samples()
        self._paths = self.get_paths()
        self.mapping_leafs = self.get_mapping_leafs()

    def __str__(self) -> str:
        """Return the tree as string."""
        repr = ""
        for node in self.nodes():
            repr += f"{node}\n"
        return repr[:-1]

    def get_mapping_leafs(self) -> dict:
        """Returns mapping between leafs and their number."""
        mapping_leafs = {}
        for leaf in self.nodes():
            if isinstance(leaf, Leaf):
                mapping_leafs[leaf.node_name] = leaf
        return mapping_leafs

    def get_documents_leafs(self) -> dict:
        """Returns mapping between documents ids and leafs and vice versa."""
        documents_to_leafs, leafs_to_documents = (
            collections.defaultdict(list),
            collections.defaultdict(list),
        )
        for node in self.nodes():
            if isinstance(node, Leaf):
                for document in node.documents:
                    documents_to_leafs[document].append(node.node_name)
                    leafs_to_documents[node.node_name].append(document)

        return dict(documents_to_leafs), dict(leafs_to_documents)

    def get_paths(self) -> list[torch.Tensor]:
        """Map leafs to their nodes."""
        self.paths.cache_clear()
        paths = collections.defaultdict(list)
        for node in self.nodes():
            if isinstance(node, Leaf):
                leaf = node
                for _ in range(leaf.level):
                    node = self.get_parent(node_name=node.node_name)
                    if node.level != 0:
                        paths[leaf.node_name].append(node.node_name)
                paths[leaf.node_name].reverse()
        return dict(paths)

    @lru_cache(maxsize=1000)
    def paths(self, leaf: int) -> dict:
        return copy.deepcopy(x=self._paths[leaf])

    def get_negative_samples(self) -> dict:
        """Return negative samples build from the tree."""
        levels = collections.defaultdict(list)
        for node in self.nodes():
            if node.node_name != self.node_name:
                levels[node.level].append((node.node_name, node.parent))
        negatives = {}
        for _, nodes in levels.items():
            for node, node_parent in nodes:
                negatives[node] = [
                    negative_node
                    for (negative_node, negative_node_parent) in nodes
                    if (negative_node != node) and (negative_node_parent == node_parent)
                ]
        return negatives

    def parameters(self) -> Generator:
        """Return the parameters of the tree."""
        for node in self.nodes():
            if isinstance(node, Leaf):
                continue
            yield node.nodes_embeddings

    def nodes(
        self,
        node: Node | Leaf = None,
    ) -> Generator:
        """Iterate over the nodes of the tree."""
        if node is None:
            node = self.tree
            yield node

        for node in node.childs:
            yield node

            if not isinstance(node, Leaf):
                yield from self.nodes(node=node)

    def get_parent(self, node_name: int | str) -> int | str:
        """Get parent nodes of a specifc node.

        Parameters
        ----------
        number:
            Number of the node.
        """
        for node in self.nodes():
            if isinstance(node, Leaf):
                continue

            for child in node.childs:
                if child.node_name == node_name:
                    return node

        return None

    @torch.no_grad()
    def __call__(
        self,
        queries: list[str],
        k: int = 100,
        k_leafs: int = 1,
        leafs: list[int] | None = None,
        score_documents: bool = True,
        beam_search_depth: int = 1,
        queries_embeddings: torch.Tensor | np.ndarray | dict = None,
        batch_size: int = 32,
        tqdm_bar: bool = True,
    ) -> tuple[torch.Tensor, list, list]:
        """Search for the closest embedding.

        Parameters
        ----------
        queries:
            Queries to search for.
        embeddings:
            Embeddings to search for.
        k:
            Number of leafs to search for.
        leafs:
            Leaf to search for.
        score_documents:
            Weather to score documents or not.
        """
        if queries_embeddings is None:
            queries_embeddings = self.scoring.transform_queries(
                queries=queries,
                batch_size=batch_size,
                tqdm_bar=tqdm_bar,
            )

        tree_scores = self._search(
            queries=queries,
            queries_embeddings=queries_embeddings,
            k_leafs=k_leafs,
            leafs=leafs,
            beam_search_depth=beam_search_depth,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
        )

        if not leafs:
            leafs_scores = [
                {
                    leaf: score.item()
                    for leaf, score in sorted(
                        query_scores.items(),
                        key=lambda item: item[1].item(),
                        reverse=True,
                    )
                    if leaf in self.mapping_leafs
                }
                for query_scores in tree_scores
            ]
        else:
            leafs_scores = [
                {leaf: query_scores[leaf].item()}
                for leaf, query_scores in zip(leafs, tree_scores)
            ]

        # We may not have k leafs for each query, so we take the minimum.
        if k_leafs > 1:
            k_leafs = min(
                min([len(query_leafs_scores) for query_leafs_scores in leafs_scores]),
                k_leafs,
            )

        candidates = {
            "leafs": np.array(
                object=[
                    list(query_leafs_scores.keys())[:k_leafs]
                    for query_leafs_scores in leafs_scores
                ]
            ),
            "scores": np.array(
                object=[
                    list(query_leafs_scores.values())[:k_leafs]
                    for query_leafs_scores in leafs_scores
                ]
            ),
            "tree_scores": tree_scores,
        }

        if not score_documents or not self.create_retrievers:
            return candidates

        if self.scoring.distinct_documents_encoder:
            queries_embeddings = self.scoring.encode_queries_for_retrieval(
                queries=queries,
            )

        leafs_queries, leafs_embeddings = (
            collections.defaultdict(list),
            collections.defaultdict(list),
        )
        for query, (query_leafs, embedding) in enumerate(
            iterable=zip(candidates["leafs"], queries_embeddings)
        ):
            for leaf in query_leafs:
                leafs_queries[leaf].append(query)
                leafs_embeddings[leaf].append(embedding)

        documents = collections.defaultdict(list)
        for leaf in leafs_queries:
            leaf_documents = self.mapping_leafs[leaf](
                queries_embeddings={
                    query: embedding
                    for query, embedding in enumerate(iterable=leafs_embeddings[leaf])
                },
                k=k,
            )
            for query, query_documents in zip(leafs_queries[leaf], leaf_documents):
                documents[query].extend(query_documents)

        # Sort documents if k > 1
        candidates["documents"] = [
            sorted(
                documents[query],
                key=lambda document: document["similarity"],
                reverse=True,
            )[: min(k, len(documents[query]))]
            if k_leafs > 1
            else documents[query]
            for query in range(len(queries))
        ]

        return candidates

    def empty(self) -> "Tree":
        """Empty the tree."""
        for node in self.nodes():
            if isinstance(node, Leaf):
                node.empty()
        return self

    def _search(
        self,
        queries: list[str],
        k_leafs: int = 1,
        leafs: list[int] | None = None,
        beam_search_depth: int = 1,
        queries_embeddings: torch.Tensor | np.ndarray | dict = None,
        batch_size: int = 32,
        tqdm_bar: bool = True,
    ) -> tuple[torch.Tensor, list, list]:
        """Search for the closest embedding with gradient.

        Parameters
        ----------
        queries:
            Queries to search for.
        embeddings:
            Embeddings to search for.
        """
        if queries_embeddings is None:
            queries_embeddings = self.scoring.transform_queries(
                queries=queries,
                batch_size=batch_size,
                tqdm_bar=tqdm_bar,
            )

        queries_embeddings = self.scoring.convert_to_tensor(
            embeddings=queries_embeddings, device=self.device
        )

        paths = (
            [(leaf, copy.copy(self.paths(leaf=leaf))) for leaf in leafs]
            if leafs is not None
            else None
        )

        tree_scores = self.tree.search(
            queries=[index for index, _ in enumerate(iterable=queries)],
            queries_embeddings=queries_embeddings,
            scoring=self.scoring,
            k=k_leafs,
            paths=paths,
            beam_search_depth=beam_search_depth,
        )

        return list(tree_scores.values())

    @torch.no_grad()
    def add(
        self,
        documents: list,
        documents_embeddings: np.ndarray | sparse.csr_matrix | dict = None,
        k: int = 1,
        documents_to_leafs: dict = None,
        batch_size: int = 32,
        tqdm_bar: bool = True,
    ) -> "Tree":
        """Add documents to the tree.

        Parameters
        ----------
        documents:
            Documents to add to the tree.
        embeddings:
            Embeddings of the documents.
        k:
            Number of leafs to add the documents to.
        """
        if documents_embeddings is None:
            documents_embeddings = self.scoring.transform_documents(
                documents=documents,
                batch_size=batch_size,
                tqdm_bar=tqdm_bar,
            )

        if documents_to_leafs is None:
            leafs = self(
                queries=[document[self.key] for document in documents],
                queries_embeddings=documents_embeddings,
                k=k,
                score_documents=False,
                tqdm_bar=False,
            )["leafs"].tolist()
        else:
            leafs = [documents_to_leafs[document[self.key]] for document in documents]

        documents_leafs, embeddings_leafs = (
            collections.defaultdict(list),
            collections.defaultdict(dict),
        )

        for document, embedding, document_leafs in zip(
            documents, documents_embeddings, leafs
        ):
            for leaf in document_leafs:
                documents_leafs[leaf].append(document)
                embeddings_leafs[leaf][document[self.key]] = embedding

        for leaf, embeddings in embeddings_leafs.items():
            self.mapping_leafs[leaf].add(
                documents=documents_leafs[leaf],
                documents_embeddings=None
                if self.scoring.distinct_documents_encoder
                else embeddings_leafs[leaf],
                scoring=self.scoring,
            )

        self.documents_to_leafs, self.leafs_to_documents = self.get_documents_leafs()
        self.negative_samples = self.get_negative_samples()
        return self

    def loss(
        self,
        queries: list[str],
        documents: list[dict],
        batch_size: int = 32,
    ) -> None:
        """Computes the loss of the tree given the input batch.

        Parameters
        ----------
        queries_embeddings:
            Embeddings of the queries.
        documents:
            Documents ids that where added to the tree.
        """
        leafs = [
            random.choice(seq=list(self.documents_to_leafs[document[self.key]]))
            for document in documents
        ]

        tree_scores = self._search(
            queries=queries,
            k_leafs=1,
            leafs=leafs,
            batch_size=batch_size,
            tqdm_bar=False,
        )

        loss, size = 0, 0
        cross_entropy = torch.nn.CrossEntropyLoss()
        for leaf, query_scores in zip(leafs, tree_scores):
            query_level_scores = [query_scores[leaf]]

            for negative_node in self.negative_samples[leaf]:
                query_level_scores.append(query_scores[negative_node])

            query_level_scores = torch.stack(
                tensors=query_level_scores, dim=0
            ).unsqueeze(dim=0)

            size += 1
            loss += cross_entropy(
                query_level_scores,
                torch.zeros(
                    query_level_scores.shape[0],
                    device=self.device,
                    dtype=torch.long,
                ),
            )

            for node in copy.copy(self.paths(leaf=leaf)):
                query_level_scores = [query_scores[node]]
                for negative_node in self.negative_samples[node]:
                    query_level_scores.append(query_scores[negative_node])

                query_level_scores = torch.stack(
                    tensors=query_level_scores, dim=0
                ).unsqueeze(dim=0)

                size += 1
                loss += cross_entropy(
                    query_level_scores,
                    torch.zeros(
                        query_level_scores.shape[0],
                        device=self.device,
                        dtype=torch.long,
                    ),
                )

        return loss / size

    def to_json(self) -> dict:
        """Return the tree as a graph."""
        return {self.node_name: self.tree.to_json()}
