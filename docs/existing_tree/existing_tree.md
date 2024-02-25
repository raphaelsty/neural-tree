# Build an index from an existing tree

Neural-Tree can build a tree from an existing graph. This is useful when we have a specific use case where we want to retrieve the right leaf for a query. 

The tree we want to pass should follow some rules:

- We should avoid nodes with a lot of children. The more children a node has, the more time it will take to explore this node.

- A node must have only one parent. This is a rule for the tree to be a tree. You can somehow duplicate a node to have it in multiple places in the tree.

Let's create a tree which has one root node, two children nodes and two leafs nodes which contains up to 3 documents.

```python
graph = {
    "root": {
        "science": {
            "machine learning": [
                {"id": 0, "content": "bayern football team"},
                {"id": 1, "content": "toulouse rugby team"},
            ],
            "computer": [
                {"id": 2, "content": "Apple Macintosh"},
                {"id": 3, "content": "Microsoft Windows"},
                {"id": 4, "content": "Linux Ubuntu"},
            ],
        },
        "history": {
            "france": [
                {"id": 5, "content": "history of france"},
                {"id": 6, "content": "french revolution"},
            ],
            "italia": [
                {"id": 7, "content": "history of rome"},
                {"id": 8, "content": "history of venice"},
            ],
        },
    }
}
```

We can now initialize either a TfIdf, a SentenceTransformer or a ColBERT tree using the graph we have created.

```python
from neural_tree import trees
from neural_cherche import models

model = models.ColBERT(
    model_name_or_path="raphaelsty/neural-cherche-colbert",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

tree = trees.ColBERT(
    key="id",  
    on=["content"],
    model=model,
    graph=graph,
    n_jobs=-1, 
)

print(tree)
```

This will output:

```python
 node root
	 node science
		 leaf computer
		 leaf machine learning
	 node history
		 leaf france
		 leaf italia
```

Once we have created our tree we can export it back to json using the `tree.to_json()`:

```python
{
    "root": {
        "science": {
            "computer": [{"id": 2}, {"id": 3}, {"id": 4}],
            "machine learning": [{"id": 0}, {"id": 1}],
        },
        "history": {"france": [{"id": 5}, {"id": 6}], "italia": [{"id": 7}, {"id": 8}]},
    }
}
```
