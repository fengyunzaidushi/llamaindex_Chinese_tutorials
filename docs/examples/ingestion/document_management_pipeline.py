#!/usr/bin/env python
# coding: utf-8

# 
# 
# Attaching a `docstore` to the ingestion pipeline will enable document management.
# 
# Using the `document.doc_id` or `node.ref_doc_id` as a grounding point, the ingestion pipeline will actively look for duplicate documents.
# 
# It works by
# - Storing a map of `doc_id` -> `document_hash`
# - If a duplicate `doc_id` is detected, and the hash has changed, the document will be re-processed
# - If the hash has not changed, the document will be skipped in the pipeline
# 
# If we do not attach a vector store, we can only check for and remove duplicate inputs.
# 
# If a vector store is attached, we can also handle upserts! We have [another guide](/examples/ingestion/redis_ingestion_pipeline.ipynb) for upserts and vector stores.

# ## Create Seed Data

# Make some test data
#('mkdir -p data')
#('echo "This is a test file: one!" > data/test1.txt')
#('echo "This is a test file: two!" > data/test2.txt')

from llama_index import SimpleDirectoryReader

# load documents with deterministic IDs
documents = SimpleDirectoryReader("./data", filename_as_id=True).load_data()

# ## Create Pipeline with Document Store

from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.ingestion import IngestionPipeline
from llama_index.storage.docstore import (
    SimpleDocumentStore,
    RedisDocumentStore,
    MongoDocumentStore,
)
from llama_index.text_splitter import SentenceSplitter

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    docstore=SimpleDocumentStore(),
)

nodes = pipeline.run(documents=documents)

print(f"Ingested {len(nodes)} Nodes")

# ### [Optional] Save/Load Pipeline
# 
# Saving the pipeline will save both the internal cache and docstore.
# 
# **NOTE:** If you were using remote caches/docstores, this step is not needed

pipeline.persist("./pipeline_storage")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ]
)

# restore the pipeline
pipeline.load("./pipeline_storage")

# ## Test the Document Management
# 
# Here, we can create a new document, as well as edit an existing document, to test the document management.
# 
# Both the new document and edited document will be ingested, while the unchanged document will be skipped

#('echo "This is a test file: three!" > data/test3.txt')
#('echo "This is a NEW test file: one!" > data/test1.txt')

documents = SimpleDirectoryReader("./data", filename_as_id=True).load_data()

nodes = pipeline.run(documents=documents)

print(f"Ingested {len(nodes)} Nodes")

# Lets confirm which nodes were ingested:

for node in nodes:
    print(f"Node: {node.text}")

# We can also verify the docstore has only three documents tracked

print(len(pipeline.docstore.docs))

