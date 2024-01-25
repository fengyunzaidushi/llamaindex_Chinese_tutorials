#!/usr/bin/env python
# coding: utf-8

# # Redis Ingestion Pipeline
# 
# This walkthrough shows how to use Redis for both the vector store, cache, and docstore in an Ingestion Pipeline.

# ## Dependencies
# 

#('pip install redis')

#('docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest')

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# ## Create Seed Data

# Make some test data
#('rm -rf test_redis_data')
#('mkdir -p test_redis_data')
#('echo "This is a test file: one!" > test_redis_data/test1.txt')
#('echo "This is a test file: two!" > test_redis_data/test2.txt')

from llama_index import SimpleDirectoryReader

# load documents with deterministic IDs
documents = SimpleDirectoryReader(
    "./test_redis_data", filename_as_id=True
).load_data()

# ## Run the Redis-Based Ingestion Pipeline
# 
# With a vector store attached, the pipeline will handle upserting data into your vector store.
# 
# However, if you only want to handle duplcates, you can change the strategy to `DUPLICATES_ONLY`.

from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.ingestion.cache import RedisCache
from llama_index.storage.docstore import RedisDocumentStore
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores import RedisVectorStore

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        embed_model,
    ],
    docstore=RedisDocumentStore.from_host_and_port(
        "localhost", 6379, namespace="document_store"
    ),
    vector_store=RedisVectorStore(
        index_name="redis_vector_store",
        index_prefix="vectore_store",
        redis_url="redis://localhost:6379",
    ),
    cache=IngestionCache(
        cache=RedisCache.from_host_and_port("localhost", 6379),
        collection="redis_cache",
    ),
    docstore_strategy=DocstoreStrategy.UPSERTS,
)

nodes = pipeline.run(documents=documents)
print(f"Ingested {len(nodes)} Nodes")

# ## Confirm documents are ingested
# 
# We can create a vector index using our vector store, and quickly ask which documents are seen.

from llama_index import VectorStoreIndex, ServiceContext

service_context = ServiceContext.from_defaults(embed_model=embed_model)

index = VectorStoreIndex.from_vector_store(
    pipeline.vector_store, service_context=service_context
)

print(
    index.as_query_engine(similarity_top_k=10).query(
        "What documents do you see?"
    )
)

# ## Add data and Ingest
# 
# Here, we can update an existing file, as well as add a new one!

#('echo "This is a test file: three!" > test_redis_data/test3.txt')
#('echo "This is a NEW test file: one!" > test_redis_data/test1.txt')

documents = SimpleDirectoryReader(
    "./test_redis_data", filename_as_id=True
).load_data()

nodes = pipeline.run(documents=documents)

print(f"Ingested {len(nodes)} Nodes")

index = VectorStoreIndex.from_vector_store(
    pipeline.vector_store, service_context=service_context
)

response = index.as_query_engine(similarity_top_k=10).query(
    "What documents do you see?"
)

print(response)

for node in response.source_nodes:
    print(node.get_text())

# As we can see, the data was deduplicated and upserted correctly! Only three nodes are in the index, even though we ran the full pipeline twice.
