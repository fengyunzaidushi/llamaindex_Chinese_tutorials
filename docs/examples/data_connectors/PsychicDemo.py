#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/data_connectors/PsychicDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Psychic Reader
# Demonstrates the Psychic data connector. Used to query data from many SaaS tools from a single LlamaIndex-compatible API.
# 
# ## Prerequisites
# Connections must first be established from the Psychic dashboard or React hook before documents can be loaded. Refer to https://docs.psychic.dev/ for more info.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import SummaryIndex, PsychicReader
from IPython.#display import Markdown, #display

# Get Psychic API key from https://dashboard.psychic.dev/api-keys
psychic_key = "PSYCHIC_API_KEY"
# Connector ID and Account ID are typically set programatically based on the application state.
account_id = "ACCOUNT_ID"
connector_id = "notion"
documents = PsychicReader(psychic_key=psychic_key).load_data(
    connector_id=connector_id, account_id=account_id
)

# set Logging to DEBUG for more detailed outputs
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
index = SummaryIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is Psychic's privacy policy?")
#display(Markdown(f"<b>{response}</b>"))

