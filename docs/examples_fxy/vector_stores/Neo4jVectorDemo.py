#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/Neo4jVectorDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Neo4j vector store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import openai

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
openai.api_key = os.environ["OPENAI_API_KEY"]

# #

from llama_index.vector_stores import Neo4jVectorStore

username = "neo4j"
password = "pleaseletmein"
url = "bolt://localhost:7687"
embed_dim = 1536

neo4j_vector = Neo4jVectorStore(username, password, url, embed_dim)

# ## Load documents, build the VectorStoreIndex

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from IPython.#display import Markdown, #display

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

from llama_index.storage.storage_context import StorageContext

storage_context = StorageContext.from_defaults(vector_store=neo4j_vector)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What happened at interleaf?")
#display(Markdown(f"<b>{response}</b>"))

# ## Hybrid search
# 
# Hybrid search uses a combination of keyword and vector search

neo4j_vector_hybrid = Neo4jVectorStore(
    username, password, url, embed_dim, hybrid_search=True
)

storage_context = StorageContext.from_defaults(
    vector_store=neo4j_vector_hybrid
)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
query_engine = index.as_query_engine()
response = query_engine.query("What happened at interleaf?")
#display(Markdown(f"<b>{response}</b>"))

# ## Load existing vector index
# 

# 
# - index_name: name of the existing vector index (default is `vector`)
# - text_node_property: name of the property that containt the text value (default is `text`)

index_name = "existing_index"
text_node_property = "text"
existing_vector = Neo4jVectorStore(
    username,
    password,
    url,
    embed_dim,
    index_name=index_name,
    text_node_property=text_node_property,
)

loaded_index = VectorStoreIndex.from_vector_store(existing_vector)

# ## Customizing responses
# 
# You can customize the retrieved information from the knowledge graph using the `retrieval_query` parameter.
# 
# The retrieval query must return the following four columns:
# 
# * text:str - The text of the returned document
# * score:str - similarity score
# * id:str - node id
# * metadata: Dict - dictionary with additional metadata (must contain `_node_type` and `_node_content` keys)

retrieval_query = (
    "RETURN 'Interleaf hired Tomaz' AS text, score, node.id AS id, "
    "{author: 'Tomaz', _node_type:node._node_type, _node_content:node._node_content} AS metadata"
)
neo4j_vector_retrieval = Neo4jVectorStore(
    username, password, url, embed_dim, retrieval_query=retrieval_query
)

loaded_index = VectorStoreIndex.from_vector_store(
    neo4j_vector_retrieval
).as_query_engine()
response = loaded_index.query("What happened at interleaf?")
#display(Markdown(f"<b>{response}</b>"))

