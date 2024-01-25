#!/usr/bin/env python
# coding: utf-8

# # Building a Live RAG Pipeline over Google Drive Files
# 
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/ingestion/ingestion_gdrive.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 

# 
# This pipeline will index Google Drive files and dump them to a Redis vector store. Afterwards, every time you rerun the ingestion pipeline, the pipeline will propagate **incremental updates**, so that only changed documents are updated in the vector store. This means that we don't re-index all the documents!
# 
# We use the following [data source](https://drive.google.com/drive/folders/1RFhr3-KmOZCR5rtp4dlOMNl3LKe1kOA5?usp=sharing) - you will need to copy these files and upload them to your own Google Drive directory! 
# 
# **NOTE**: You will also need to setup a service account and credentials.json. See our LlamaHub page for the Google Drive loader for more details: https://llamahub.ai/l/google_drive
# 
# 

# ## Setup
# 
# We install required packages and launch the Redis Docker image.

#('pip install llama-hub pydrive redis docx2txt transformers torch')

# if creating a new container
#('docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest')
# # if starting an existing container
# !docker start -a redis-stack

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# ## Define Ingestion Pipeline
# 
# Here we define the ingestion pipeline. Given a set of documents, we will run sentence splitting/embedding transformations, and then load them into a Redis docstore/vector store.
# 
# The vector store is for indexing the data + storing the embeddings, the docstore is for tracking duplicates.

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

vector_store = RedisVectorStore(
    index_name="redis_vector_store",
    index_prefix="vectore_store",
    redis_url="redis://localhost:6379",
)

cache = IngestionCache(
    cache=RedisCache.from_host_and_port("localhost", 6379),
    collection="redis_cache",
)

# Optional: clear vector store if exists
if vector_store._index_exists():
    vector_store.delete_index()

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        embed_model,
    ],
    docstore=RedisDocumentStore.from_host_and_port(
        "localhost", 6379, namespace="document_store"
    ),
    vector_store=vector_store,
    cache=cache,
    docstore_strategy=DocstoreStrategy.UPSERTS,
)

# ### Define our Vector Store Index
# 
# We define our index to wrap the underlying vector store.

from llama_index import VectorStoreIndex, ServiceContext

service_context = ServiceContext.from_defaults(embed_model=embed_model)

index = VectorStoreIndex.from_vector_store(
    pipeline.vector_store, service_context=service_context
)

# ## Load Initial Data
# 
# Here we load data from our [Google Drive Loader](https://llamahub.ai/l/google_drive) on LlamaHub. 
# 
# The loaded docs are the header sections of our [Use Cases from our documentation](https://docs.llamaindex.ai/en/latest/use_cases/q_and_a.html).

from llama_hub.google_drive.base import GoogleDriveReader

loader = GoogleDriveReader()

def load_data(folder_id: str):
    docs = loader.load_data(folder_id=folder_id)
    for doc in docs:
        doc.id_ = doc.metadata["file_name"]
    return docs

docs = load_data(folder_id="1RFhr3-KmOZCR5rtp4dlOMNl3LKe1kOA5")
# print(docs)

nodes = pipeline.run(documents=docs)
print(f"Ingested {len(nodes)} Nodes")

# Since this is our first time starting up the vector store, we see that we've transformed/ingested all the documents into it (by chunking, and then by embedding).

# ### Ask Questions over Initial Data

query_engine = index.as_query_engine()

response = query_engine.query("What are the sub-types of question answering?")

print(str(response))

# ## Modify and Reload the Data
# 
# Let's try modifying our ingested data! 
# 
# We modify the "Q&A" doc to include an extra "structured analytics" block of text. See our [updated document](https://docs.google.com/document/d/1QQMKNAgyplv2IUOKNClEBymOFaASwmsZFoLmO_IeSTw/edit?usp=sharing) as a reference.
# 
# Now let's rerun the ingestion pipeline.

docs = load_data(folder_id="1RFhr3-KmOZCR5rtp4dlOMNl3LKe1kOA5")
nodes = pipeline.run(documents=docs)
print(f"Ingested {len(nodes)} Nodes")

# Notice how only one node is ingested. This is beacuse only one document changed, while the other documents stayed the same. This means that we only need to re-transform and re-embed one document!

# ### Ask Questions over New Data

query_engine = index.as_query_engine()

response = query_engine.query("What are the sub-types of question answering?")

print(str(response))

