
<div align="center">
  <h1>Neural-Tree</h1>
  <p>Neural Search</p>
</div>

<p align="center"><img width=500 src="docs/img/neural_tree.png"/></p>

<div align="center">
  <!-- Documentation -->
  <a href="https://raphaelsty.github.io/neural-tree/"><img src="https://img.shields.io/website?label=Documentation&style=flat-square&url=https%3A%2F%2Fraphaelsty.github.io/neural-tree/%2F" alt="documentation"></a>
  <!-- License -->
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="license"></a>
</div>

Neural-Tree is designed to accelerate inference of Information Retrieval models. Neural-Tree implements SIGIR 2023 [Li et al., 2023](https://dl.acm.org/doi/10.1145/3539618.3591651) publication. 

Neural-Tree is tailored to work efficiently with ColBERT, Sentence Transformer and TfIdf models.

We can create a tree from scratch or use an existing tree structure. The tree must be trained with a set of paired queries and documents. Once the tree is trained, we can retrieve relevant documents or leafs from the tree given a set of queries.

## Installation

We can install neural-tree using:

```
pip install neural-tree
```

If we plan to evaluate our model while training install:

```
pip install "neural-tree[eval]"
```

## Documentation

The complete documentation is available [here](https://raphaelsty.github.io/neural-tree/).


## Quick Start

In order to create a tree-based index, we will need to gather training data.
The training data consists of queries and documents paired:

```python
train_queries = [
    "query document a",
    "query document b",
    "query document c",
    "query document d",
]

train_documents = [
    {"id": "doc a", "text": "document a"},
    {"id": "doc b", "text": "document b"},
    {"id": "doc c", "text": "document c"},
    {"id": "doc d", "text": "document d"},
]
```

The following code shows how to train a tree model using the `scifact` dataset.
You can replace the `scifact` dataset with any other dataset.

```python
import torch

from nlp_tree import clustering, datasets, trees, utils
from neural_cherche import models

documents, train_queries, train_documents = datasets.load_beir_train(
    dataset_name="scifact",
)

model = models.ColBERT(
    model_name_or_path="raphaelsty/neural-cherche-colbert",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

tree = trees.ColBERT(
    key="id", # The field to use as a key for the documents.
    on=["title", "text"], # The fields to use for the model.
    model=model,
    documents=documents, 
    leaf_balance_factor=100, # Minimum number of documents per leaf.
    branch_balance_factor=5, # Number of childs per node.
    n_jobs=-1, # We want to set it to 1 when using Google Colab, -1 otherwise.
)

optimizer = torch.optim.AdamW(lr=3e-2, params=list(tree.parameters()))

for step, batch_queries, batch_documents in utils.iter(
    queries=train_queries,
    documents=train_documents,
    shuffle=True,
    epochs=50,
    batch_size=1024,
):
    loss = tree.loss(
        queries=batch_queries,
        documents=batch_documents,
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

documents, queries_ids, test_queries, qrels = datasets.load_beir_test(
    dataset_name="scifact",
)

documents_to_leafs = clustering.optimize_leafs(
    tree=tree,
    queries=train_queries + test_queries,
    documents=documents,
)

tree = tree.add(
    documents=documents,
    documents_to_leafs=documents_to_leafs,
)
```

## Search

```python
candidates = tree(
    queries=test_queries,
    k_leafs=2, # number of leafs to search
    k=10, # number of documents to retrieve
)
```

## Evaluation

We can evaluate the performance of the tree using the following code:

```python
documents, queries_ids, test_queries, qrels = datasets.load_beir_test(
    dataset_name="scifact",
)

candidates = tree(
    queries=test_queries,
    k_leafs=2,
    k=10,
)

scores = utils.evaluate(
    scores=candidates["documents"],
    qrels=qrels,
    queries_ids=queries_ids,
)

print(scores)
```

## Benchmarks


## References

- [Constructing Tree-based Index for Efficient and Effective Dense Retrieval, Li et al., 2023](https://github.com/cshaitao/jtr)

