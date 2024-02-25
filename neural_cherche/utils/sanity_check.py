__all__ = ["sanity_check"]


def sanity_check(
    branch_balance_factor: int, leaf_balance_factor: int, graph: dict, documents: list
) -> None:
    """Check if the input is valid."""
    if branch_balance_factor < 2:
        raise ValueError("Branch balance factor must be greater than 1.")

    if leaf_balance_factor < 1:
        raise ValueError("Leaf balance factor must be greater than 0.")

    if graph is not None:
        if len(graph.keys()) > 1:
            raise ValueError("Graph must have only one root node.")

    if documents is None and graph is None:
        raise ValueError("You must provide either documents or an existing graph.")
