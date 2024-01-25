#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/customization/streaming/SimpleIndexDemo-streaming.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Streaming

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Load documents, build the VectorStoreIndex

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

index = VectorStoreIndex.from_documents(documents)

# #### Query Index

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
response_stream = query_engine.query(
    "What did the author do growing up?",
)

response_stream.print_response_stream()

