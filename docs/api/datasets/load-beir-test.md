# load_beir_test

Load BEIR testing dataset.



## Parameters

- **dataset_name** (*str*)

    Dataset name.



## Examples

```python
>>> from neural_tree import datasets

>>> documents, queries_ids, queries, qrels = datasets.load_beir_test(
...     dataset_name="scifact",
... )

>>> len(documents)
5183

>>> assert len(queries_ids) == len(queries) == len(qrels)
```

