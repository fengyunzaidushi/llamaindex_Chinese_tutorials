#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/data_connectors/DiscordDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Discord Reader
# Demonstrates our Discord data connector

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# This is due to the fact that we use asyncio.loop_until_complete in
# the DiscordReader. Since the Jupyter kernel itself runs on
# an event loop, we need to add some help with nesting
#('pip install nest_asyncio')
import nest_asyncio

nest_asyncio.apply()

from llama_index import SummaryIndex, DiscordReader
from IPython.#display import Markdown, #display
import os

discord_token = os.getenv("DISCORD_TOKEN")
channel_ids = [1057178784895348746]  # Replace with your channel_id
documents = DiscordReader(discord_token=discord_token).load_data(
    channel_ids=channel_ids
)

index = SummaryIndex.from_documents(documents)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")

#display(Markdown(f"<b>{response}</b>"))

