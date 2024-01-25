#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/ingestion/advanced_ingestion_pipeline.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#('pip install llama-index')

# # Advanced Ingestion Pipeline
# 

# 
# - MongoDB transformation caching
# - Automatic vector databse insertion
# - A custom transformation 

# ## Redis Cache Setup
# 
# All node + transformation combinations will have their outputs cached, which will save time on duplicate runs.

from llama_index.ingestion.cache import RedisCache, IngestionCache

ingest_cache = IngestionCache(
    cache=RedisCache.from_host_and_port(host="127.0.0.1", port=6379),
    collection="my_test_cache",
)

# ## Vector DB Setup
# 
# For this example, we use weaviate as a vector store.

#('pip install weaviate-client')

import weaviate

auth_config = weaviate.AuthApiKey(api_key="...")

client = weaviate.Client(url="https://...", auth_client_secret=auth_config)

from llama_index.vector_stores import WeaviateVectorStore

vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="CachingTest"
)

# ## Transformation Setup

from llama_index.text_splitter import TokenTextSplitter
from llama_index.embeddings import HuggingFaceEmbedding

text_splitter = TokenTextSplitter(chunk_size=512)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ### Custom Transformation

import re
from llama_index.schema import TransformComponent

class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
        return nodes

# ## Running the pipeline

from llama_index.ingestion import IngestionPipeline

pipeline = IngestionPipeline(
    transformations=[TextCleaner(), text_splitter, embed_model],
    vector_store=vector_store,
    cache=ingest_cache,
)

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader("../data/paul_graham/").load_data()

nodes = pipeline.run(documents=documents)

# ## Using our populated vector store

import os

# needed for the LLM in the query engine
os.environ["OPENAI_API_KEY"] = "sk-..."

from llama_index import VectorStoreIndex, ServiceContext

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    service_context=ServiceContext.from_defaults(embed_model=embed_model),
)

query_engine = index.as_query_engine()

print(query_engine.query("What did the author do growing up?"))

# ## Re-run Ingestion to test Caching
# 
# The next code block will execute almost instantly due to caching.

pipeline = IngestionPipeline(
    transformations=[TextCleaner(), text_splitter, embed_model],
    cache=ingest_cache,
)

nodes = pipeline.run(documents=documents)

# ## Clear the cache

ingest_cache.clear()

