# ColBERT

ColBERT retriever.



## Parameters

- **key** (*str*)

- **on** (*str | list[str]*)

- **device** (*str*)




## Methods

???- note "__call__"

    Rank documents  givent queries.

    **Parameters**

    - **queries_embeddings**     (*dict[str, torch.Tensor]*)    
    - **batch_size**     (*int*)     – defaults to `32`    
    - **k**     (*int*)     – defaults to `None`    
    - **tqdm_bar**     (*bool*)     – defaults to `False`    
    
???- note "add"

    Add documents embeddings.

    **Parameters**

    - **documents_embeddings**     (*dict*)    
    
???- note "encode_documents"

    Encode documents.

    **Parameters**

    - **documents**     (*list[str]*)    
    - **batch_size**     (*int*)     – defaults to `32`    
    - **tqdm_bar**     (*bool*)     – defaults to `True`    
    - **query_mode**     (*bool*)     – defaults to `False`    
    - **kwargs**    
    
???- note "encode_queries"

    Encode queries.

    **Parameters**

    - **queries**     (*list[str]*)    
    - **batch_size**     (*int*)     – defaults to `32`    
    - **tqdm_bar**     (*bool*)     – defaults to `True`    
    - **query_mode**     (*bool*)     – defaults to `True`    
    - **kwargs**    
    
