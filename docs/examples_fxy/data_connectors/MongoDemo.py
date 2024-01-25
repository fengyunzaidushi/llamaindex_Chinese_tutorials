#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/data_connectors/MongoDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # MongoDB Reader
# Demonstrates our MongoDB data connector

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index import SummaryIndex, SimpleMongoReader
from IPython.#display import Markdown, #display
import os

host = "<host>"
port = "<port>"
db_name = "<db_name>"
collection_name = "<collection_name>"
# query_dict is passed into db.collection.find()
query_dict = {}
field_names = ["text"]
reader = SimpleMongoReader(host, port)
documents = reader.load_data(
    db_name, collection_name, field_names, query_dict=query_dict
)

index = SummaryIndex.from_documents(documents)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")

#display(Markdown(f"<b>{response}</b>"))

