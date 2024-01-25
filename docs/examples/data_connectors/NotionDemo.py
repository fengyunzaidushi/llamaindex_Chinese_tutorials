#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/data_connectors/NotionDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Notion Reader
# Demonstrates our Notion data connector

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index import SummaryIndex, NotionPageReader
from IPython.#display import Markdown, #display
import os

integration_token = os.getenv("NOTION_INTEGRATION_TOKEN")
page_ids = ["<page_id>"]
documents = NotionPageReader(integration_token=integration_token).load_data(
    page_ids=page_ids
)

index = SummaryIndex.from_documents(documents)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")

#display(Markdown(f"<b>{response}</b>"))

# You can also pass the id of a database to index all the pages in that database:

database_id = "<database-id>"

# https://developers.notion.com/docs/working-with-databases for how to find your database id

documents = NotionPageReader(integration_token=integration_token).load_data(
    database_id=database_id
)

print(documents)

# set Logging to DEBUG for more detailed outputs
index = SummaryIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")
#display(Markdown(f"<b>{response}</b>"))

