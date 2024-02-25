# BaseScore

Base class for scoring functions.






## Methods

???- note "convert_to_tensor"

    Transform sparse matrix to tensor.

    **Parameters**

    - **embeddings**     (*scipy.sparse._csr.csr_matrix | numpy.ndarray*)    
    - **device**     (*str*)    
    
???- note "distinct_documents_encoder"

    Return True if the encoder is distinct for documents and nodes.

    
???- note "encode_queries_for_retrieval"

    Encode queries for retrieval.

    **Parameters**

    - **queries**     (*list[str]*)    
    
???- note "get_retriever"

    Create a retriever

    
???- note "leaf_scores"

    Return the scores of the embeddings.

    **Parameters**

    - **queries_embeddings**     (*torch.Tensor*)    
    - **leaf_embedding**     (*torch.Tensor*)    
    
???- note "nodes_scores"

    Score between queries and nodes embeddings.

    **Parameters**

    - **queries_embeddings**     (*torch.Tensor*)    
    - **nodes_embeddings**     (*torch.Tensor*)    
    
???- note "stack"

    Stack list of embeddings.

    - **embeddings**     (*list[scipy.sparse._csr.csr_matrix | numpy.ndarray | dict]*)    
    
???- note "transform_documents"

    Transform documents to embeddings.

    **Parameters**

    - **documents**     (*list[dict]*)    
    
???- note "transform_queries"

    Transform queries to embeddings.

    **Parameters**

    - **queries**     (*list[str]*)    
    
