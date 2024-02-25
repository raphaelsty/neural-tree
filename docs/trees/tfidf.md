# TfIdf

To create a tree-based index for Sentence Transformers, we will need to:
 
- Gather the whole set of documents we want to index.
- Gather queries paired to documents.
- Sample the training set in order to evaluate the index.

```python
# Whole set of documents we want to index.
documents = [
    {"id": 0, "content": "paris"},
    {"id": 1, "content": "london"},
    {"id": 2, "content": "berlin"},
    {"id": 3, "content": "rome"},
    {"id": 4, "content": "bordeaux"},
    {"id": 5, "content": "milan"},    
]

# Paired training documents
train_documents = [
    {"id": 0, "content": "paris"},
    {"id": 1, "content": "london"},
    {"id": 2, "content": "berlin"},
    {"id": 3, "content": "rome"},
]

# Paired training queries
train_queries = [
    "paris is the capital of france",
    "london is the capital of england",
    "berlin is the capital of germany",
    "rome is the capital of italy",
]
```

Let's train the index using the `documents`, `train_queries` and `train_documents` we have gathered.

```python
import torch
from neural_tree import trees, utils
from sentence_transformers import SentenceTransformer

tree = trees.TfIdf(
    key="id",  # The field to use as a key for the documents.
    on=["title", "content"],  # The fields to use for the model.
    documents=documents,
    leaf_balance_factor=100,  # Minimum number of documents per leaf.
    branch_balance_factor=5,  # Number of childs per node.
    n_jobs=-1,  # We want to set it to 1 when using Google Colab.
)

optimizer = torch.optim.AdamW(lr=3e-3, params=list(tree.parameters()))

for step, batch_queries, batch_documents in utils.iter(
    queries=train_queries,
    documents=train_documents,
    shuffle=True,
    epochs=50,
    batch_size=32,
):
    loss = tree.loss(
        queries=batch_queries,
        documents=batch_documents,
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

We can already use the `tree` to search for documents using the `tree` method.

The `call` method of the tree outputs a dictionary containing several key pieces of information: the retrieved leaves under leafs, the score assigned to each leaf under scores, a record of the explored nodes and leaves along with their scores under tree_scores, and the documents retrieved for each query listed under documents.

```python
tree(
    queries=["history"],
    k=10, # Number of documents to return for each query. 
    k_leafs=1, # The number of leafs to return for each query.
)
```

```python
{
    "leafs": array([["10"]], dtype="<U2"), # leafs retrieved
    "scores": array([[1.79485011]]), # scores for each leaf
    "tree_scores": [ # history of nodes and leafs explored with the respective scores
        {
            "10": tensor(1.7949),
            "12": tensor(1.5722),
            "14": tensor(1.5132),
            "13": tensor(0.9872),
            "11": tensor(0.8582),
        }
    ],
    "documents": [ # documents retrieved for each query
        [
            {"id": 3, "similarity": 1.9020360708236694, "leaf": "10"},
            {"id": 2, "similarity": 1.5113722085952759, "leaf": "10"},
        ]
    ],
}
```

Once we have trained our index, we should further optimize the tree by duplicating some documents in the tree's leafs. This will allow us to have a better recall when searching for documents. We can use the `clustering.optimize_leafs` method to optimize the tree. We shoud gather as much queries as possible to optimize the tree.

```python
from neural_tree import clustering

test_queries = [
    "bordeaux is a city in the west of france",
    "milan is a city in the north of italy",
]

documents_to_leafs = clustering.optimize_leafs(
    tree=tree,
    queries=train_queries + test_queries,  # We gather all the queries we have.
    documents=documents,  # The whole set of documents we want to index.
)

tree = tree.add(
    documents=documents,
    documents_to_leafs=documents_to_leafs,
)
```

We can now use the optimized `tree` to search for documents. We can pass one or multiple queries.

```python
tree(
    queries=["history"],
    k=10, # The number of documents to return for each query. 
    k_leafs=1, # The number of leafs to return for each query.
)
```
