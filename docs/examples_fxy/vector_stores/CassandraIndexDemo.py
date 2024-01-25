#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/CassandraIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Cassandra Vector Store

# [Apache Cassandra®](https://cassandra.apache.org) is a NoSQL, row-oriented, highly scalable and highly available database. Starting with version 5.0, the database ships with [vector search](https://cassandra.apache.org/doc/trunk/cassandra/vector-search/overview.html) capabilities.
# 
# DataStax [Astra DB through CQL](https://docs.datastax.com/en/astra-serverless/docs/vector-search/quickstart.html) is a managed serverless database built on Cassandra, offering the same interface and strengths.
# 
# **This notebook shows the basic usage of the Cassandra Vector Store in LlamaIndex.**
# 
# To run the full code you need either a running Cassandra cluster equipped with Vector 
# Search capabilities or a DataStax Astra DB instance.

# ## Setup

#('pip install --quiet "astrapy>=0.5.8"')

import os
from getpass import getpass

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext,
)
from llama_index.vector_stores import CassandraVectorStore

# The next step is to initialize CassIO with a global DB connection: this is the only step that is done slightly differently for a Cassandra cluster and Astra DB:

# ##

# as described in the [Cassandra driver documentation](https://docs.datastax.com/en/developer/python-driver/latest/api/cassandra/cluster/#module-cassandra.cluster).
# The details vary (e.g. with network settings and authentication), but this might be something like:

from cassandra.cluster import Cluster

cluster = Cluster(["127.0.0.1"])
session = cluster.connect()

import cassio

CASSANDRA_KEYSPACE = input("CASSANDRA_KEYSPACE = ")

cassio.init(session=session, keyspace=CASSANDRA_KEYSPACE)

# ##

# 
# - the Database ID, e.g. 01234567-89ab-cdef-0123-456789abcdef
# - the Token, e.g. AstraCS:6gBhNmsk135.... (it must be a "Database Administrator" token)
# - Optionally a Keyspace name (if omitted, the default one for the database will be used)

ASTRA_DB_ID = input("ASTRA_DB_ID = ")
ASTRA_DB_TOKEN = getpass("ASTRA_DB_TOKEN = ")

desired_keyspace = input("ASTRA_DB_KEYSPACE (optional, can be left empty) = ")
if desired_keyspace:
    ASTRA_DB_KEYSPACE = desired_keyspace
else:
    ASTRA_DB_KEYSPACE = None

import cassio

cassio.init(
    database_id=ASTRA_DB_ID,
    token=ASTRA_DB_TOKEN,
    keyspace=ASTRA_DB_KEYSPACE,
)

# ### OpenAI key
# 

os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key:")

# ### Download data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ## Creating and populating the Vector Store
# 
# You will now load some essays by Paul Graham from a local file and store them into the Cassandra Vector Store.

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
print(f"Total documents: {len(documents)}")
print(f"First document, id: {documents[0].doc_id}")
print(f"First document, hash: {documents[0].hash}")
print(
    "First document, text"
    f" ({len(documents[0].text)} characters):\n{'='*20}\n{documents[0].text[:360]} ..."
)

# ##
# 
# Creation of the vector store entails creation of the underlying database table if it does not exist yet:

cassandra_store = CassandraVectorStore(
    table="cass_v_table", embedding_dimension=1536
)

# Now wrap this store into an `index` LlamaIndex abstraction for later querying:

storage_context = StorageContext.from_defaults(vector_store=cassandra_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# Note that the above `from_documents` call does several things at once: it splits the input documents into chunks of manageable size ("nodes"), computes embedding vectors for each node, and stores them all in the Cassandra Vector Store.

# ## Querying the store

# ### Basic querying

query_engine = index.as_query_engine()
response = query_engine.query("Why did the author choose to work on AI?")
print(response.response)

# ### MMR-based queries
# 
# The MMR (maximal marginal relevance) method is designed to fetch text chunks from the store that are at the same time relevant to the query but as different as possible from each other, with the goal of providing a broader context to the building of the final answer:

query_engine = index.as_query_engine(vector_store_query_mode="mmr")
response = query_engine.query("Why did the author choose to work on AI?")
print(response.response)

# ## Connecting to an existing store
# 
# Since this store is backed by Cassandra, it is persistent by definition. So, if you want to connect to a store that was created and populated previously, here is how:

new_store_instance = CassandraVectorStore(
    table="cass_v_table", embedding_dimension=1536
)

# Create index (from preexisting stored vectors)
new_index_instance = VectorStoreIndex.from_vector_store(
    vector_store=new_store_instance
)

# now you can do querying, etc:
query_engine = new_index_instance.as_query_engine(similarity_top_k=5)
response = query_engine.query(
    "What did the author study prior to working on AI?"
)

print(response.response)

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

new_store_instance.delete(nodes_with_scores[0].node.ref_doc_id)

# Repeat the very same query and check the results now. You should see _no results_ being found:

nodes_with_scores = retriever.retrieve(
    "What did the author study prior to working on AI?"
)

print(f"Found {len(nodes_with_scores)} nodes.")

# ## Metadata filtering
# 
# The Cassandra vector store support metadata filtering in the form of exact-match `key=value` pairs at query time. The following cells, which work on a brand new Cassandra table, demonstrate this feature.
# 

md_storage_context = StorageContext.from_defaults(
    vector_store=CassandraVectorStore(
        table="cass_v_table_md", embedding_dimension=1536
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
    "./data/paul_graham", file_metadata=my_file_metadata
).load_data()
md_index = VectorStoreIndex.from_documents(
    md_documents, storage_context=md_storage_context
)

# 
# 
# That's it: you can now add filtering to your query engine:

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

md_query_engine = md_index.as_query_engine(
    filters=MetadataFilters(
        filters=[ExactMatchFilter(key="source_type", value="essay")]
    )
)
md_response = md_query_engine.query(
    "did the author appreciate Lisp and painting?"
)
print(md_response.response)

# To test that the filtering is at play, try to change it to use only `"dinos"` documents... there will be no answer this time :)
