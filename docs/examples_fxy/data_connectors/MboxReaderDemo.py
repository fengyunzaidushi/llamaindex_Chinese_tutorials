#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/data_connectors/MboxReaderDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Mbox Reader

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

get_ipython().run_line_magic('env', 'OPENAI_API_KEY=sk-************')

from llama_index import MboxReader, VectorStoreIndex

documents = MboxReader().load_data(
    "mbox_data_dir", max_count=1000
)  # Returns list of documents

index = VectorStoreIndex.from_documents(
    documents
)  

query_engine = index.as_query_engine()
res = query_engine.query("When did i have that call with the London office?")

res.response

