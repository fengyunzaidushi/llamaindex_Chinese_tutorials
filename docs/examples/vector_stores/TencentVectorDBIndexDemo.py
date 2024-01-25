#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/TencentVectorDBIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Tencent Cloud VectorDB

# >[Tencent Cloud VectorDB](https://cloud.tencent.com/document/product/1709) is a fully managed, self-developed, enterprise-level distributed database service designed for storing, retrieving, and analyzing multi-dimensional vector data. The database supports multiple index types and similarity calculation methods. A single index can support a vector scale of up to 1 billion and can support millions of QPS and millisecond-level query latency. Tencent Cloud Vector Database can not only provide an external knowledge base for large models to improve the accuracy of large model responses but can also be widely used in AI fields such as recommendation systems, NLP services, computer vision, and intelligent customer service.
# 
# **This notebook shows the basic usage of TencentVectorDB as a Vector Store in LlamaIndex.**
# 
# To run, you should have a [Database instance.](https://cloud.tencent.com/document/product/1709/95101)

# ## Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

#('pip install tcvectordb')

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores import TencentVectorDB
from llama_index.vector_stores.tencentvectordb import (
    CollectionParams,
    FilterField,
)
import tcvectordb

tcvectordb.debug.DebugEnable = False

# ### Please provide OpenAI access key
# 

import openai

OPENAI_API_KEY = getpass.getpass("OpenAI API Key:")
openai.api_key = OPENAI_API_KEY

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ## Creating and populating the Vector Store
# 
# You will now load some essays by Paul Graham from a local file and store them into the Tencent Cloud VectorDB.

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()
print(f"Total documents: {len(documents)}")
print(f"First document, id: {documents[0].doc_id}")
print(f"First document, hash: {documents[0].hash}")
print(
    f"First document, text ({len(documents[0].text)} characters):\n{'='*20}\n{documents[0].text[:360]} ..."
)

# ##
# 
# Creation of the vector store entails creation of the underlying database collection if it does not exist yet:

vector_store = TencentVectorDB(
    url="http://10.0.X.X",
    key="eC4bLRy2va******************************",
    collection_params=CollectionParams(dimension=1536, drop_exists=True),
)

# Now wrap this store into an `index` LlamaIndex abstraction for later querying:

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# Note that the above `from_documents` call does several things at once: it splits the input documents into chunks of manageable size ("nodes"), computes embedding vectors for each node, and stores them all in the Tencent Cloud VectorDB.

# ## Querying the store

# ### Basic querying

query_engine = index.as_query_engine()
response = query_engine.query("Why did the author choose to work on AI?")
print(response)

# ### MMR-based queries
# 
# The MMR (maximal marginal relevance) method is designed to fetch text chunks from the store that are at the same time relevant to the query but as different as possible from each other, with the goal of providing a broader context to the building of the final answer:

query_engine = index.as_query_engine(vector_store_query_mode="mmr")
response = query_engine.query("Why did the author choose to work on AI?")
print(response)

# ## Connecting to an existing store
# 
# Since this store is backed by Tencent Cloud VectorDB, it is persistent by definition. So, if you want to connect to a store that was created and populated previously, here is how:

new_vector_store = TencentVectorDB(
    url="http://10.0.X.X",
    key="eC4bLRy2va******************************",
    collection_params=CollectionParams(dimension=1536, drop_exists=False),
)

# Create index (from preexisting stored vectors)
new_index_instance = VectorStoreIndex.from_vector_store(
    vector_store=new_vector_store
)

# now you can do querying, etc:
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query(
    "What did the author study prior to working on AI?"
)

print(response)

# ## Removing documents from the index
# 
# First get an explicit list of pieces of a document, or "nodes", from a `Retriever` spawned from the index:

retriever = new_index_instance.as_retriever(
    vector_store_query_mode="mmr",
    similarity_top_k=3,
    vector_store_kwargs={"mmr_prefetch_factor": 4},
)
nodes_with_scores = retriever.retrieve(
    "What did the author study prior to working on AI?"
)

print(f"Found {len(nodes_with_scores)} nodes.")
for idx, node_with_score in enumerate(nodes_with_scores):
    print(f"    [{idx}] score = {node_with_score.score}")
    print(f"        id    = {node_with_score.node.node_id}")
    print(f"        text  = {node_with_score.node.text[:90]} ...")

# But wait! When using the vector store, you should consider the **document** as the sensible unit to delete, and not any individual node belonging to it. Well, in this case, you just inserted a single text file, so all nodes will have the same `ref_doc_id`:

print("Nodes' ref_doc_id:")
print("\n".join([nws.node.ref_doc_id for nws in nodes_with_scores]))

# Now let's say you need to remove the text file you uploaded:

new_vector_store.delete(nodes_with_scores[0].node.ref_doc_id)

# Repeat the very same query and check the results now. You should see _no results_ being found:

nodes_with_scores = retriever.retrieve(
    "What did the author study prior to working on AI?"
)

print(f"Found {len(nodes_with_scores)} nodes.")

# ## Metadata filtering
# 
# The Tencent Cloud VectorDB vector store support metadata filtering in the form of exact-match `key=value` pairs at query time. The following cells, which work on a brand new collection, demonstrate this feature.
# 

filter_fields = [
    FilterField(name="source_type"),
]

md_storage_context = StorageContext.from_defaults(
    vector_store=TencentVectorDB(
        url="http://10.0.X.X",
        key="eC4bLRy2va******************************",
        collection_params=CollectionParams(
            dimension=1536, drop_exists=True, filter_fields=filter_fields
        ),
    )
)

def my_file_metadata(file_name: str):
    """Depending on the input file name, associate a different metadata."""
    if "essay" in file_name:
        source_type = "essay"
    elif "dinosaur" in file_name:
        # this (unfortunately) will not happen in this demo
        source_type = "dinos"
    else:
        source_type = "other"
    return {"source_type": source_type}

# Load documents and build index
md_documents = SimpleDirectoryReader(
    "../data/paul_graham", file_metadata=my_file_metadata
).load_data()
md_index = VectorStoreIndex.from_documents(
    md_documents, storage_context=md_storage_context
)

# That's it: you can now add filtering to your query engine:

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

md_query_engine = md_index.as_query_engine(
    filters=MetadataFilters(
        filters=[ExactMatchFilter(key="source_type", value="essay")]
    )
)
md_response = md_query_engine.query(
    "How long it took the author to write his thesis?"
)
print(md_response.response)

# To test that the filtering is at play, try to change it to use only `"dinos"` documents... there will be no answer this time :)
