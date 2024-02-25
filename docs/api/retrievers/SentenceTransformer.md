# SentenceTransformer

Sentence Transformer retriever.



## Parameters

- **key** (*str*)

- **device** (*str*) – defaults to `cpu`



## Examples

```python
>>> from neural_tree import retrievers
>>> from sentence_transformers import SentenceTransformer
>>> from pprint import pprint

>>> model = SentenceTransformer("all-mpnet-base-v2")

>>> retriever = retrievers.SentenceTransformer(key="id")

>>> retriever = retriever.add(
...     documents_embeddings={
...         0: model.encode("Paris is the capital of France."),
...         1: model.encode("Berlin is the capital of Germany."),
...         2: model.encode("Paris and Berlin are European cities."),
...         3: model.encode("Paris and Berlin are beautiful cities."),
...     }
... )

>>> queries_embeddings = {
...     0: model.encode("Paris"),
...     1: model.encode("Berlin"),
... }

>>> candidates = retriever(queries_embeddings=queries_embeddings, k=2)
>>> pprint(candidates)
[[{'id': 0, 'similarity': 0.644777984318611},
  {'id': 3, 'similarity': 0.52865785276988}],
 [{'id': 1, 'similarity': 0.6901492368348436},
  {'id': 3, 'similarity': 0.5457692206973245}]]
```

## Methods

???- note "__call__"

    Retrieve documents.

    **Parameters**

    - **queries_embeddings**     (*dict[int, numpy.ndarray]*)    
    - **k**     (*int | None*)     – defaults to `100`    
    - **kwargs**    
    
???- note "add"

    Add documents to the faiss index.

    **Parameters**

    - **documents_embeddings**     (*dict[int, numpy.ndarray]*)    
    
