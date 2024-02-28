# Evaluate

Neural-tree evaluation is based on [RANX](https://github.com/AmenRa/ranx). We can also download datasets of [BEIR Benchmark](https://github.com/beir-cellar/beir) with the `utils.load_beir` function.


## Installation

```bash
pip install "neural-tree[eval]"
```

## Usage

Here is an example of how to train a tree-based index using the `scifact` dataset and how to evaluate it.

```python
import torch
from neural_cherche import models
from sentence_transformers import SentenceTransformer

from neural_tree import clustering, datasets, trees, utils

documents, train_queries, train_documents = datasets.load_beir_train(
    dataset_name="scifact",
)


model = models.ColBERT(
    model_name_or_path="raphaelsty/neural-cherche-colbert",
    device="cuda",
)

# We intialize a ColBERT index from a
# SentenceTransformer-based hierarchical clustering.
tree = trees.ColBERT(
    key="id",
    on=["title", "text"],
    model=model,
    sentence_transformer=SentenceTransformer(model_name_or_path="all-mpnet-base-v2"),
    documents=documents,
    leaf_balance_factor=100,
    branch_balance_factor=5,
    n_jobs=-1,
    device="cuda",
    faiss_device="cuda",
)

optimizer = torch.optim.AdamW(lr=3e-3, params=list(tree.parameters()))


for step, batch_queries, batch_documents in utils.iter(
    queries=train_queries,
    documents=train_documents,
    shuffle=True,
    epochs=50,
    batch_size=128,
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

candidates = tree(
    queries=test_queries,
    k_leafs=2,  # number of leafs to search
    k=10,  # number of documents to retrieve
)

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

```python
{"ndcg@10": 0.6957728027724698, "hits@1": 0.59, "hits@2": 0.69, "hits@3": 0.76, "hits@4": 0.8133333333333334, "hits@5": 0.8533333333333334, "hits@10": 0.91}
```

## Evaluation dataset

Here are what documents should looks like (an id with multiples fields):

```python
[
    {
        "id": "document_0",
        "title": "Bayesian measures of model complexity and fit",
        "text": "Summary. We consider the problem of comparing complex hierarchical models in which the number of parameters is not clearly defined. Using an information theoretic argument we derive a measure pD for the effective number of parameters in a model as the difference between the posterior mean of the deviance and the deviance at the posterior means of the parameters of interest. In general pD approximately corresponds to the trace of the product of Fisher's information and the posterior covariance, which in normal models is the trace of the ‘hat’ matrix projecting observations onto fitted values. Its properties in exponential families are explored. The posterior mean deviance is suggested as a Bayesian measure of fit or adequacy, and the contributions of individual observations to the fit and complexity can give rise to a diagnostic plot of deviance residuals against leverages. Adding pD to the posterior mean deviance gives a deviance information criterion for comparing models, which is related to other information criteria and has an approximate decision theoretic justification. The procedure is illustrated in some examples, and comparisons are drawn with alternative Bayesian and classical proposals. Throughout it is emphasized that the quantities required are trivial to compute in a Markov chain Monte Carlo analysis.",
    },
    {
        "id": "document_1",
        "title": "Simplifying likelihood ratios",
        "text": "Likelihood ratios are one of the best measures of diagnostic accuracy, although they are seldom used, because interpreting them requires a calculator to convert back and forth between “probability” and “odds” of disease. This article describes a simpler method of interpreting likelihood ratios, one that avoids calculators, nomograms, and conversions to “odds” of disease. Several examples illustrate how the clinician can use this method to refine diagnostic decisions at the bedside.",
    },
]
```

Queries is a list of strings:

```python
[
    "Varenicline monotherapy is more effective after 12 weeks of treatment compared to combination nicotine replacement therapies with varenicline or bupropion.",
    "Venules have a larger lumen diameter than arterioles.",
    "Venules have a thinner or absent smooth layer compared to arterioles.",
    "Vitamin D deficiency effects the term of delivery.",
    "Vitamin D deficiency is unrelated to birth weight.",
    "Women with a higher birth weight are more likely to develop breast cancer later in life.",
]
```

QueriesIds is a list of ids with respect to the order of queries:

```python
[
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
]
```

Qrels is the mapping between queries ids as key and dict of relevant documents with 1 as value:

```python
{
    "1": {"document_0": 1},
    "3": {"document_10": 1},
    "5": {"document_5": 1},
    "13": {"document_22": 1},
    "36": {"document_23": 1, "document_0": 1},
    "42": {"document_2": 1},
}
```

## Metrics

We can evaluate our model with various metrics detailed [here](https://amenra.github.io/ranx/metrics/).