# TfIdf

TfIdf retriever



## Parameters

- **key** (*str*)

- **on** (*list[str]*)




## Methods

???- note "__call__"

    Retrieve documents from batch of queries.

    **Parameters**

    - **queries_embeddings**     (*dict[str, scipy.sparse._csr.csr_matrix]*)    
    - **k**     (*int*)     – defaults to `None`    
    - **batch_size**     (*int*)     – defaults to `2000`    
    - **tqdm_bar**     (*bool*)     – defaults to `True`    
    
???- note "add"

    Add new documents to the TFIDF retriever. The tfidf won't be refitted.

    **Parameters**

    - **documents_embeddings**     (*dict[str, scipy.sparse._csr.csr_matrix]*)    
    
???- note "encode_documents"

    Encode queries into sparse matrix.

    **Parameters**

    - **documents**     (*list[dict]*)    
    - **model**     (*sklearn.feature_extraction.text.TfidfVectorizer*)    
    
???- note "encode_queries"

    Encode queries into sparse matrix.

    **Parameters**

    - **queries**     (*list[str]*)    
    - **model**     (*sklearn.feature_extraction.text.TfidfVectorizer*)    
    
???- note "top_k"

    Return the top k documents for each query.

    **Parameters**

    - **similarities**     (*scipy.sparse._csc.csc_matrix*)    
    - **k**     (*int*)    
    
