
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

Neural-Tree is a library designed to accelerate inference of information retrieval pipelines. Neural-Tree implements SIGIR 2023 publication [Constructing Tree-based Index for Efficient and Effective Dense Retrieval, Li et al., 2023](https://dl.acm.org/doi/10.1145/3539618.3591651). 

Neural-Tree is tailored to work efficiently with token-level embeddings such as ColBERT. You can create a tree from scratch or use an existing tree. The tree must be trained with a set of paired queries and documents. Once the tree is trained, you can use it to retrieve documents.

You can ask the tree to retrieve documents or to retrieve the leafs that are most relevant to a query.

Neural-Tree is compatible with ColBERT, Sentence Transformers and TF-IDF models.

## Installation

To install `neural-tree` run:

```bash
pip install neural-tree
```

If you want to install the evaluation dependencies, you can use the following command:

```bash
pip install "neural-tree[eval]"
```

## Training

In order to create a tree-based index, we will need to gather training data.
The training data consists of queries and documents paired. The following code shows a simple training dataset where each query is paired with a document.

```python
train_queries = [
    "query a",
    "query b",
    "query c",
    "query d",
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

from neural_tree import clustering, datasets, trees, utils
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
    branch_balance_factor=3, # Number of childs per node.
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

