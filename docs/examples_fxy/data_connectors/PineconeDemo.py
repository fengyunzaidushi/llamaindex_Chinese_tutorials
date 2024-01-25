#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/data_connectors/PineconeDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Pinecone Reader

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

api_key = "<api_key>"

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.readers.pinecone import PineconeReader

reader = PineconeReader(api_key=api_key, environment="us-west1-gcp")

# the id_to_text_map specifies a mapping from the ID specified in Pinecone to your text.
id_to_text_map = {
    "id1": "text blob 1",
    "id2": "text blob 2",
}

# the query_vector is an embedding representation of your query_vector
# Example query vector:
#   query_vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

query_vector = [n1, n2, n3, ...]

# NOTE: Required args are index_name, id_to_text_map, vector.

# See the API reference: https://docs.pinecone.io/reference/query
# and also the Python client: https://github.com/pinecone-io/pinecone-python-client
# for more details.
documents = reader.load_data(
    index_name="quickstart",
    id_to_text_map=id_to_text_map,
    top_k=3,
    vector=query_vector,
    separate_documents=True,
)

# ### Create index 

index = SummaryIndex.from_documents(documents)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")

#display(Markdown(f"<b>{response}</b>"))

