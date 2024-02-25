# load_beir_train

Load training dataset.



## Parameters

- **dataset_name** (*str*)

    Dataset name



## Examples

```python
>>> from neural_tree import datasets

>>> documents, train_queries, train_documents = datasets.load_beir_train(
...     dataset_name="scifact",
... )

>>> len(documents)
5183

>>> assert len(train_queries) == len(train_documents)
```

