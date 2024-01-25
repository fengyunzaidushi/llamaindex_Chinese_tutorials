#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/data_connectors/DashvectorReaderDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # DashVector Reader

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

api_key = os.environ["DASHVECTOR_API_KEY"]

from llama_index.readers import DashVectorReader

reader = DashVectorReader(api_key=api_key)

import numpy as np

# the id_to_text_map specifies a mapping from the ID specified in DashVector to your text.
id_to_text_map = {
    "id1": "text blob 1",
    "id2": "text blob 2",
}

# the query_vector is an embedding representation of your query_vector
query_vector = [n1, n2, n3, ...]

# NOTE: Required args are index_name, id_to_text_map, vector.

# See the Python client: https://pypi.org/project/dashvector/ for more details.
documents = reader.load_data(
    collection_name="quickstart",
    id_to_text_map=id_to_text_map,
    top_k=3,
    vector=query_vector,
    filter="key = 'value'",
)

# ### Create index 

from llama_index.indices import ListIndex
from IPython.#display import Markdown, #display

index = ListIndex.from_documents(documents)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")

#display(Markdown(f"<b>{response}</b>"))

