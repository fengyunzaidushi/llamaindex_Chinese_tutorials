#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/data_connectors/DeepLakeReader.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # DeepLake Reader

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import getpass
import os
import random
import textwrap

from llama_index import VectorStoreIndex
from llama_index.readers.deeplake import DeepLakeReader

os.environ["OPENAI_API_KEY"] = getpass.getpass("open ai api key: ")

reader = DeepLakeReader()
query_vector = [random.random() for _ in range(1536)]
documents = reader.load_data(
    query_vector=query_vector,
    dataset_path="hub://activeloop/paul_graham_essay",
    limit=5,
)

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What was a hard moment for the author?")
print(textwrap.fill(str(response), 100))

