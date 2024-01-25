#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/ElasticsearchIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Elasticsearch Vector Store

# Elasticsearch is a distributed, RESTful search and analytics engine, capable of performing both vector and keyword search. It is built on top of the Apache Lucene library.
# 
# [Signup](https://cloud.elastic.co/registration?utm_source=llama-index&utm_content=documentation) for a free trial.
# 
# Requires Elasticsearch 8.9.0 or higher and AIOHTTP.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# ## Running and connecting to Elasticsearch
# Two ways to setup an Elasticsearch instance for use with:
# 
# ### Elastic Cloud
# Elastic Cloud is a managed Elasticsearch service. [Signup](https://cloud.elastic.co/registration?utm_source=llama-index&utm_content=documentation) for a free trial.
# 
# ### Locally
# Get started with Elasticsearch by running it locally. The easiest way is to use the official Elasticsearch Docker image. See the Elasticsearch Docker documentation for more information.
# 
# ```bash
# docker run -p 9200:9200 \
#   -e "discovery.type=single-node" \
#   -e "xpack.security.enabled=false" \
#   -e "xpack.security.http.ssl.enabled=false" \
#   -e "xpack.license.self_generated.type=trial" \
#   docker.elastic.co/elasticsearch/elasticsearch:8.9.0
# ```
# 
# ## Configuring ElasticsearchStore
# The ElasticsearchStore class is used to connect to an Elasticsearch instance. It requires the following parameters:
# 
#         - index_name: Name of the Elasticsearch index. Required.
#         - es_client: Optional. Pre-existing Elasticsearch client.
#         - es_url: Optional. Elasticsearch URL.
#         - es_cloud_id: Optional. Elasticsearch cloud ID.
#         - es_api_key: Optional. Elasticsearch API key.
#         - es_user: Optional. Elasticsearch username.
#         - es_password: Optional. Elasticsearch password.
#         - text_field: Optional. Name of the Elasticsearch field that stores the text.
#         - vector_field: Optional. Name of the Elasticsearch field that stores the
#                     embedding.
#         - batch_size: Optional. Batch size for bulk indexing. Defaults to 200.
#         - distance_strategy: Optional. Distance strategy to use for similarity search.
#                         Defaults to "COSINE".
# 
# ### Example: Connecting locally
# ```python
# from llama_index.vector_stores import ElasticsearchStore
# 
# es = ElasticsearchStore(
#     index_name="my_index",
#     es_url="http://localhost:9200",
# )
# ```
# 
# ### Example: Connecting to Elastic Cloud with username and password
# 
# ```python
# from llama_index.vector_stores import ElasticsearchStore
# 
# es = ElasticsearchStore(
#     index_name="my_index",
#     es_cloud_id="<cloud-id>", # found within the deployment page
#     es_user="elastic"
#     es_password="<password>" # provided when creating deployment. Alternatively can reset password.
# )
# ```
# 
# ### Example: Connecting to Elastic Cloud with API Key
# 
# ```python
# from llama_index.vector_stores import ElasticsearchStore
# 
# es = ElasticsearchStore(
#     index_name="my_index",
#     es_cloud_id="<cloud-id>", # found within the deployment page
#     es_api_key="<api-key>" # Create an API key within Kibana (Security -> API Keys)
# )
# ```
# 

# #### Load documents, build VectorStoreIndex with Elasticsearch

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import ElasticsearchStore

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# initialize without metadata filter
from llama_index.storage.storage_context import StorageContext

vector_store = ElasticsearchStore(
    es_url="http://localhost:9200",
    # Or with Elastic Cloud
    # es_cloud_id="my_cloud_id",
    # es_user="elastic",
    # es_password="my_password",
    index_name="paul_graham",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# ## Basic Example
# We are going to ask the query engine a question about the data we just indexed.

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("what were his investments in Y Combinator?")
print(response)

# ## Metadata Filters
# Here we are going to index a few documents with metadata so that we can apply filters to the query engine.

from llama_index.schema import TextNode

nodes = [
    TextNode(
        text="The Shawshank Redemption",
        metadata={
            "author": "Stephen King",
            "theme": "Friendship",
        },
    ),
    TextNode(
        text="The Godfather",
        metadata={
            "director": "Francis Ford Coppola",
            "theme": "Mafia",
        },
    ),
    TextNode(
        text="Inception",
        metadata={
            "director": "Christopher Nolan",
        },
    ),
]

# initialize the vector store
vector_store_metadata_example = ElasticsearchStore(
    index_name="movies_metadata_example",
    es_url="http://localhost:9200",
)
storage_context = StorageContext.from_defaults(
    vector_store=vector_store_metadata_example
)
index = VectorStoreIndex(nodes, storage_context=storage_context)

# Metadata filter
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)

retriever = index.as_retriever(filters=filters)

retriever.retrieve("What is inception about?")

# ## Custom Filters and overriding Query 
# llama-index supports ExactMatchFilters only at the moment. Elasticsearch supports a wide range of filters, including range filters, geo filters, and more. To use these filters, you can pass them in as a list of dictionaries to the `es_filter` parameter.

def custom_query(query, query_str):
    print("custom query", query)
    return query

query_engine = index.as_query_engine(
    vector_store_kwargs={
        "es_filter": [{"match": {"content": "growing up"}}],
        "custom_query": custom_query,
    }
)
response = query_engine.query("what were his investments in Y Combinator?")
print(response)

