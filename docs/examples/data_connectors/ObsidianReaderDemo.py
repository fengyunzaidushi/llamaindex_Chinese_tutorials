#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/data_connectors/ObsidianReaderDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Obsidian Reader

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

get_ipython().run_line_magic('env', 'OPENAI_API_KEY=sk-************')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import ObsidianReader, VectorStoreIndex

documents = ObsidianReader(
    "/Users/hursh/vault"
).load_data()  # Returns list of documents

index = VectorStoreIndex.from_documents(
    documents
)  

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
res = query_engine.query("What is the meaning of life?")

res.response

