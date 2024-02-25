# ColBERT

TfIdf scoring function.



## Parameters

- **key** (*str*)

- **on** (*list | str*)

- **documents** (*list*)

- **model** (*neural_cherche.models.colbert.ColBERT*) – defaults to `None`

- **device** (*str*) – defaults to `cpu`

- **kwargs**


## Attributes

- **distinct_documents_encoder**

    Return True if the encoder is distinct for documents and nodes.


## Examples

```python
>>> from neural_tree import trees, scoring
>>> from neural_cherche import models
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from pprint import pprint
>>> import torch

>>> _ = torch.manual_seed(42)

>>> documents = [
...     {"id": 0, "text": "Paris is the capital of France."},
...     {"id": 1, "text": "Berlin is the capital of Germany."},
...     {"id": 2, "text": "Paris and Berlin are European cities."},
...     {"id": 3, "text": "Paris and Berlin are beautiful cities."},
... ]

>>> model = models.ColBERT(
...     model_name_or_path="sentence-transformers/all-mpnet-base-v2",
...     embedding_size=128,
...     max_length_document=96,
...     max_length_query=32,
... )

>>> tree = trees.ColBERTTree(
...    key="id",
...    on="text",
...    model=model,
...    documents=documents,
...    leaf_balance_factor=1,
...    branch_balance_factor=2,
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

>>> tree.leafs_to_documents
{'100': [0], '101': [1], '110': [2], '111': [3]}

>>> candidates = tree(
...    queries=["Paris is the capital of France.", "Paris and Berlin are European cities."],
...    k_leafs=2,
...    k=2,
... )

>>> candidates["scores"]
array([[28.12037659, 18.32332611],
       [29.28324509, 21.38923264]])

>>> candidates["leafs"]
array([['100', '101'],
       ['110', '111']], dtype='<U3')

>>> pprint(candidates["tree_scores"])
[{'10': tensor(28.1204),
  '100': tensor(28.1204),
  '101': tensor(18.3233),
  '11': tensor(20.9327)},
 {'10': tensor(21.6886),
  '11': tensor(29.2832),
  '110': tensor(29.2832),
  '111': tensor(21.3892)}]

>>> pprint(candidates["documents"])
[[{'id': 0, 'leaf': '100', 'similarity': 28.120376586914062},
  {'id': 1, 'leaf': '101', 'similarity': 18.323326110839844}],
 [{'id': 2, 'leaf': '110', 'similarity': 29.283245086669922},
  {'id': 3, 'leaf': '111', 'similarity': 21.389232635498047}]]
```

## Methods

???- note "average"

    Average embeddings.

    - **embeddings**     (*torch.Tensor*)    
    
???- note "convert_to_tensor"

    Transform sparse matrix to tensor.

    **Parameters**

    - **embeddings**     (*numpy.ndarray | torch.Tensor*)    
    - **device**     (*str*)    
    
???- note "encode_queries_for_retrieval"

    Encode queries for retrieval.

    **Parameters**

    - **queries**     (*list[str]*)    
    
???- note "get_retriever"

    Create a retriever

    
???- note "leaf_scores"

    Return the scores of the embeddings.

    **Parameters**

    - **queries_embeddings**     (*torch.Tensor*)    
    - **leaf_embedding**     (*torch.Tensor*)    
    
???- note "nodes_scores"

    Score between queries and nodes embeddings.

    **Parameters**

    - **queries_embeddings**     (*torch.Tensor*)    
    - **nodes_embeddings**     (*torch.Tensor*)    
    
???- note "stack"

    Stack list of embeddings.

    **Parameters**

    - **embeddings**     (*list[torch.Tensor | numpy.ndarray]*)    
    
???- note "transform_documents"

    Transform documents to embeddings.

    **Parameters**

    - **documents**     (*list[dict]*)    
    - **batch_size**     (*int*)    
    - **tqdm_bar**     (*bool*)    
    - **kwargs**    
    
???- note "transform_queries"

    Transform queries to embeddings.

    **Parameters**

    - **queries**     (*list[str]*)    
    - **batch_size**     (*int*)    
    - **tqdm_bar**     (*bool*)    
    - **kwargs**    
    
