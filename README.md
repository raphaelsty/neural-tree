
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

<p></p>

Are tree-based indexes the counterpart of standard ANN algorithms for token-level embeddings IR models? Neural-Tree replicate the SIGIR 2023 publication [Constructing Tree-based Index for Efficient and Effective Dense Retrieval](https://dl.acm.org/doi/10.1145/3539618.3591651) in order to accelerate ColBERT. Neural-Tree is compatible with Sentence Transformers and TfIdf models as in the original paper. 

Neural-Tree creates a tree using hierarchical clustering of documents and then learn embeddings in each node of the tree using paired queries and documents. Additionally, there is the flexibility to input an existing tree structure in JSON format to build the index.

The optimization of the index by Neural-Tree is geared towards maintaining the performance level of the original model while significantly speeding up the search process. It is important to note that Neural-Tree does not modify the underlying model; therefore, it is advisable to initiate tree creation with a model that has already been fine-tuned. Given that Neural-Tree does not alter the model, the index training process is relatively quick.

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

The following code shows how to train a tree model using the `scifact` dataset which load
train_documents, train_queries and the whole corpora of documents.

```python
import torch

from neural_cherche import models
from neural_tree import clustering, datasets, trees, utils

documents, train_queries, train_documents = datasets.load_beir_train(
    dataset_name="scifact",
)

model = models.ColBERT(
    model_name_or_path="raphaelsty/neural-cherche-colbert",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

tree = trees.ColBERT(
    key="id",
    on=["title", "text"],
    model=model,
    documents=documents, 
    leaf_balance_factor=100,
    branch_balance_factor=5,
    n_jobs=-1, # We want to set it to 1 when using Google Colab.
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

We can now search for documents using the tree we have trained:

```python
candidates = tree(
    queries=["science topic", "history topic"],
    k_leafs=2, # number of leafs to search
    k=10, # number of documents to retrieve
)

print(candidates)
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

<table>
<thead>
  <tr>
    <th colspan="2" rowspan="2"></th>
    <th colspan="9">Scifact Dataset</th>
  </tr>
  <tr>
    <th colspan="4">Vanilla</th>
    <th colspan="5">Neural-Tree </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>model</td>
    <td>HuggingFace Checkpoint</td>
    <td>ndcg@10</td>
    <td>hits@10</td>
    <td>hits@1</td>
    <td>queries / second</td>
    <td>ndcg@10</td>
    <td>hits@10</td>
    <td>hits@1</td>
    <td>queries / second</td>
    <td>Acceleration</td>
  </tr>
  <tr>
    <td>TfIdf<br>Cherche</td>
    <td>-</td>
    <td>0,61</td>
    <td>0,85</td>
    <td>0,47</td>
    <td>760</td>
    <td>0,56</td>
    <td>0,82</td>
    <td>0,42</td>
    <td>1080</td>
    <td>+42.11%</td>
  </tr>
  <tr>
    <td>SentenceTransformer GPU<br>Faiss.IndexFlatL2 CPU</td>
    <td>sentence-transformers/all-mpnet-base-v2</td>
    <td>0,66</td>
    <td>0,89</td>
    <td>0,53</td>
    <td>475</td>
    <td>0,66</td>
    <td>0,88</td>
    <td>0,53</td>
    <td>518</td>
    <td>+9.05%</td>
  </tr>
  <tr>
    <td>ColBERT<br>Neural-Cherche GPU</td>
    <td>raphaelsty/neural-cherche-colbert</td>
    <td>0,70</td>
    <td>0,92</td>
    <td>0,58</td>
    <td>3</td>
    <td>0,70</td>
    <td>0,91</td>
    <td>0,59</td>
    <td>256</td>
    <td>x85</td>
  </tr>
</tbody>
</table>

Note that this benchmark do not implement [ColBERTV2](https://arxiv.org/abs/2112.01488) efficient retrieval but rather compare ColBERT raw retrieval with Neural-Tree. We could accelerate SentenceTransformer vanilla by using optimized Faiss index.

## Contributing

Contributions to Neural-Tree are welcome. This tool aims to tackle tree visualization,
modeling node topics, and using the tree to speed up LLM search. We are also looking to improve clustering of ColBERT embeddings while performing hierarchical clustering which is now handled by TfIdf. The optimization clustering is also a good place to contribute in order to provide an end-to-end ColBERT cluster optimization without TfIdf.

## License

This project is licensed under the terms of the MIT license.

## References

- [Constructing Tree-based Index for Efficient and Effective Dense Retrieval, Github](https://github.com/cshaitao/jtr)

- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)

- [Myriade](https://github.com/MaxHalford/myriade)

 