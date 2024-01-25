#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/TypesenseDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Typesense Vector Store

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Load documents, build the VectorStoreIndex

# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from IPython.#display import Markdown, #display

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

from llama_index.vector_stores.typesense import TypesenseVectorStore
from typesense import Client

typesense_client = Client(
    {
        "api_key": "xyz",
        "nodes": [{"host": "localhost", "port": "8108", "protocol": "http"}],
        "connection_timeout_seconds": 2,
    }
)
typesense_vector_store = TypesenseVectorStore(typesense_client)
storage_context = StorageContext.from_defaults(
    vector_store=typesense_vector_store
)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# #### Query Index

from llama_index.schema import QueryBundle
from llama_index.embeddings import OpenAIEmbedding

# By default, typesense vector store uses vector search. You need to provide the embedding yourself.
query_str = "What did the author do growing up?"
embed_model = OpenAIEmbedding()
# If your service context has an embed_model you can also do:
# embed_model = index.service_context.embed_model
query_embedding = embed_model.get_agg_embedding_from_queries(query_str)
query_bundle = QueryBundle(query_str, embedding=query_embedding)
response = index.as_query_engine().query(query_bundle)

#display(Markdown(f"<b>{response}</b>"))

from llama_index.vector_stores.types import VectorStoreQueryMode

# You can also use text search

query_bundle = QueryBundle(query_str=query_str)
response = index.as_query_engine(
    vector_store_query_mode=VectorStoreQueryMode.TEXT_SEARCH
).query(query_bundle)
#display(Markdown(f"<b>{response}</b>"))

