# SentenceTransformer

Sentence Transformer scoring function.



## Parameters

- **key** (*str*)

- **on** (*str | list*)

- **model** (*sentence_transformers.SentenceTransformer.SentenceTransformer*)

- **device** (*str*) – defaults to `cpu`

- **faiss_device** (*str*) – defaults to `cpu`


## Attributes

- **distinct_documents_encoder**

    Return True if the encoder is distinct for documents and nodes.


## Examples

```python
>>> from neural_tree import trees, scoring
>>> from sentence_transformers import SentenceTransformer
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
...    scoring=scoring.SentenceTransformer(key="id", on=["text"], model=SentenceTransformer("all-mpnet-base-v2")),
...    leaf_balance_factor=1,
...    branch_balance_factor=2,
...    n_jobs=1,
... )

>>> print(tree)
node 1
    node 11
        node 110
            leaf 1100
            leaf 1101
        leaf 111
    leaf 10

>>> candidates = tree(
...    queries=["paris", "berlin"],
...    k_leafs=2,
... )

>>> candidates["scores"]
array([[0.72453916, 0.60635257],
       [0.58386189, 0.57546711]])

>>> candidates["leafs"]
array([['111', '10'],
       ['1101', '1100']], dtype='<U4')

>>> pprint(candidates["tree_scores"])
[{'10': tensor(0.6064),
  '11': tensor(0.7245),
  '110': tensor(0.5542),
  '1100': tensor(0.5403),
  '1101': tensor(0.5542),
  '111': tensor(0.7245)},
 {'10': tensor(0.5206),
  '11': tensor(0.5797),
  '110': tensor(0.5839),
  '1100': tensor(0.5755),
  '1101': tensor(0.5839),
  '111': tensor(0.4026)}]

>>> pprint(candidates["documents"])
[[{'id': 0, 'leaf': '111', 'similarity': 0.6447779347587058},
  {'id': 1, 'leaf': '10', 'similarity': 0.43175890864117644}],
 [{'id': 3, 'leaf': '1101', 'similarity': 0.545769273959571},
  {'id': 2, 'leaf': '1100', 'similarity': 0.54081365990618}]]
```

## Methods

???- note "average"

    Average embeddings.

    - **embeddings**     (*numpy.ndarray*)    
    
???- note "convert_to_tensor"

    Convert numpy array to torch tensor.

    **Parameters**

    - **embeddings**     (*numpy.ndarray*)    
    - **device**     (*str*)    
    
???- note "encode_queries_for_retrieval"

    Encode queries for retrieval.

    - **queries**     (*list[str]*)    
    
???- note "get_retriever"

    Create a retriever

    
???- note "leaf_scores"

    Computes scores between query and leaf embedding.

    **Parameters**

    - **queries_embeddings**     (*torch.Tensor*)    
    - **leaf_embedding**     (*torch.Tensor*)    
    
???- note "nodes_scores"

    Score between queries and nodes embeddings.

    **Parameters**

    - **queries_embeddings**     (*torch.Tensor*)    
    - **nodes_embeddings**     (*torch.Tensor*)    
    
???- note "stack"

    Stack embeddings.

    - **embeddings**     (*list[numpy.ndarray]*)    
    
???- note "transform_documents"

    Transform documents to embeddings.

    **Parameters**

    - **documents**     (*list[dict]*)    
    - **batch_size**     (*int*)    
    - **kwargs**    
    
???- note "transform_queries"

    Transform queries to embeddings.

    **Parameters**

    - **queries**     (*list[str]*)    
    - **batch_size**     (*int*)    
    - **kwargs**    
    
