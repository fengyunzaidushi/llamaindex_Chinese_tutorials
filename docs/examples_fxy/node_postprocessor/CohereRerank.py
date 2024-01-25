#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/node_postprocessor/CohereRerank.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Cohere Rerank

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    pprint_response,
)

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# build index
index = VectorStoreIndex.from_documents(documents=documents)

# #### Retrieve top 10 most relevant nodes, then filter with Cohere Rerank

import os
from llama_index.postprocessor.cohere_rerank import CohereRerank

api_key = os.environ["COHERE_API_KEY"]
cohere_rerank = CohereRerank(api_key=api_key, top_n=2)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[cohere_rerank],
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)

pprint_response(response)

# ### Directly retrieve top 2 most similar nodes

query_engine = index.as_query_engine(
    similarity_top_k=2,
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)

# Retrieved context is irrelevant and response is hallucinated.

pprint_response(response)

