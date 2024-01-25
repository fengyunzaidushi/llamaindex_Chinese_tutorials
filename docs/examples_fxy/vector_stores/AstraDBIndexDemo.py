#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/AstraDBIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Astra DB
# 
# >[DataStax Astra DB](https://docs.datastax.com/en/astra/home/astra.html) is a serverless vector-capable database built on Apache Cassandra and accessed through an easy-to-use JSON API.
# 
# To run this notebook you need a DataStax Astra DB instance running in the cloud (you can get one for free at [datastax.com](https://astra.datastax.com)).
# 
# You should ensure you have `llama-index` and `astrapy` installed:

#('pip install llama-index')
#('pip install "astrapy>=0.6.0"')

# ### Please provide database connection parameters and secrets:

import os
import getpass

api_endpoint = input(
    "\nPlease enter your Database Endpoint URL (e.g. 'https://4bc...datastax.com'):"
)

token = getpass.getpass(
    "\nPlease enter your 'Database Administrator' Token (e.g. 'AstraCS:...'):"
)

os.environ["OPENAI_API_KEY"] = getpass.getpass(
    "\nPlease enter your OpenAI API Key (e.g. 'sk-...'):"
)

# ### Import needed package dependencies:

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores import AstraDBVectorStore

# ### Load some example data:

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Read the data:

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
print(f"Total documents: {len(documents)}")
print(f"First document, id: {documents[0].doc_id}")
print(f"First document, hash: {documents[0].hash}")
print(
    "First document, text"
    f" ({len(documents[0].text)} characters):\n{'='*20}\n{documents[0].text[:360]} ..."
)

# ### Create the Astra DB Vector Store object:

astra_db_store = AstraDBVectorStore(
    token=token,
    api_endpoint=api_endpoint,
    collection_name="astra_v_table",
    embedding_dimension=1536,
)

# ### Build the Index from the Documents:

storage_context = StorageContext.from_defaults(vector_store=astra_db_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# ### Query using the index:

query_engine = index.as_query_engine()
response = query_engine.query("Why did the author choose to work on AI?")

print(response.response)

