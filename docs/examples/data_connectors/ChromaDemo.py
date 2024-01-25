#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/data_connectors/ChromaDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Chroma Reader

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.readers.chroma import ChromaReader

# The chroma reader loads data from a persisted Chroma collection.
# This requires a collection name and a persist directory.

reader = ChromaReader(
    collection_name="chroma_collection",
    persist_directory="examples/data_connectors/chroma_collection",
)

# the query_vector is an embedding representation of your query.
# Example query vector:
#   query_vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

query_vector = [n1, n2, n3, ...]

# NOTE: Required args are collection_name, query_vector.
# See the Python client: https://github.com/chroma-core/chroma
# for more details.
documents = reader.load_data(
    collection_name="demo", query_vector=query_vector, limit=5
)

# ### Create index

from llama_index.indices import SummaryIndex

index = SummaryIndex.from_documents(documents)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")

#display(Markdown(f"<b>{response}</b>"))

