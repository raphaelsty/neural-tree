# TfIdf

TfIdf scoring function.



## Parameters

- **key** (*str*)

- **on** (*list | str*)

- **documents** (*list*)

- **tfidf_nodes** (*sklearn.feature_extraction.text.TfidfVectorizer | None*) – defaults to `None`

- **tfidf_documents** (*sklearn.feature_extraction.text.TfidfVectorizer | None*) – defaults to `None`

- **kwargs**


## Attributes

- **distinct_documents_encoder**

    Return True if the encoder is distinct for documents and nodes.


## Examples

```python
>>> from neural_tree import trees, scoring
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from pprint import pprint

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
array([[0.99999994, 0.63854915],
       [0.99999994, 0.72823119]])

>>> candidates["leafs"]
array([['100', '101'],
       ['110', '111']], dtype='<U3')

>>> pprint(candidates["tree_scores"])
[{'10': tensor(1.0000),
  '100': tensor(1.0000),
  '101': tensor(0.6385),
  '11': tensor(0.1076)},
 {'10': tensor(0.1076),
  '11': tensor(1.0000),
  '110': tensor(1.0000),
  '111': tensor(0.7282)}]

>>> pprint(candidates["documents"])
[[{'id': 0, 'leaf': '100', 'similarity': 0.9999999999999978},
  {'id': 1, 'leaf': '101', 'similarity': 0.39941742405759667}],
 [{'id': 2, 'leaf': '110', 'similarity': 0.9999999999999978},
  {'id': 3, 'leaf': '111', 'similarity': 0.5385719658738707}]]
```

## Methods

???- note "average"

    Average embeddings.

    - **embeddings**     (*scipy.sparse._csr.csr_matrix*)    
    
???- note "convert_to_tensor"

    Transform sparse matrix to tensor.

    **Parameters**

    - **embeddings**     (*scipy.sparse._csr.csr_matrix*)    
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

    - **embeddings**     (*list[scipy.sparse._csr.csr_matrix]*)    
    
???- note "transform_documents"

    Transform documents to embeddings.

    **Parameters**

    - **documents**     (*list[dict]*)    
    - **kwargs**    
    
???- note "transform_queries"

    Transform queries to embeddings.

    **Parameters**

    - **queries**     (*list[str]*)    
    - **kwargs**    
    
