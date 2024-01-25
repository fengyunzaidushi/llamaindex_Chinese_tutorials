#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/data_connectors/TwitterDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Twitter Reader

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index import VectorStoreIndex, TwitterTweetReader
from IPython.#display import Markdown, #display
import os

# create an app in https://developer.twitter.com/en/apps
BEARER_TOKEN = "<bearer_token>"

# create reader, specify twitter handles
reader = TwitterTweetReader(BEARER_TOKEN)
documents = reader.load_data(["@twitter_handle1"])

index = VectorStoreIndex.from_documents(documents)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")

#display(Markdown(f"<b>{response}</b>"))

