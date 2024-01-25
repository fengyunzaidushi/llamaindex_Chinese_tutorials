#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/data_connectors/SlackDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Slack Reader
# Demonstrates our Slack data connector

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import SummaryIndex, SlackReader
from IPython.#display import Markdown, #display
import os

slack_token = os.getenv("SLACK_BOT_TOKEN")
channel_ids = ["<channel_id>"]
documents = SlackReader(slack_token=slack_token).load_data(
    channel_ids=channel_ids
)

index = SummaryIndex.from_documents(documents)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")

#display(Markdown(f"<b>{response}</b>"))

