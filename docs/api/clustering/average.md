# average

Replace KMeans clustering with average clustering when an existing graph is provided.



## Parameters

- **key** (*str*)

- **documents** (*list*)

- **documents_embeddings** (*numpy.ndarray | scipy.sparse._csr.csr_matrix*)

- **graph**

- **scoring**

- **device** (*str*)



## Examples

```python
>>> from neural_tree import clustering, scoring
>>> import numpy as np

>>> documents = [
...     {"id": 0, "text": "Paris is the capital of France."},
...     {"id": 1, "text": "Berlin is the capital of Germany."},
...     {"id": 2, "text": "Paris and Berlin are European cities."},
...     {"id": 3, "text": "Paris and Berlin are beautiful cities."},
... ]

>>> documents_embeddings = np.array([
...     [1, 1],
...     [1, 2],
...     [10, 10],
...     [1, 3],
... ])

>>> graph = {1: {11: {111: [{'id': 0}, {'id': 3}], 112: [{'id': 1}]}, 12: {121: [{'id': 2}], 122: [{'id': 3}]}}}

>>> clustering.average(
...     key="id",
...     documents_embeddings=documents_embeddings,
...     documents=documents,
...     graph=graph[1],
...     scoring=scoring.SentenceTransformer(key="id", on=["text"], model=None),
... )
```

