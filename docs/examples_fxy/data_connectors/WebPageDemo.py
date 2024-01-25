#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/data_connectors/WebPageDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Web Page Reader
# 
# Demonstrates our web page reader.

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# #### Using SimpleWebPageReader

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index import SummaryIndex, SimpleWebPageReader
from IPython.#display import Markdown, #display
import os

# NOTE: the html_to_text=True option requires html2text to be installed

documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["http://paulgraham.com/worked.html"]
)

documents[0]

index = SummaryIndex.from_documents(documents)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

#display(Markdown(f"<b>{response}</b>"))

# #### Using TrafilaturaWebReader

from llama_index import TrafilaturaWebReader

documents = TrafilaturaWebReader().load_data(
    ["http://paulgraham.com/worked.html"]
)

index = SummaryIndex.from_documents(documents)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

#display(Markdown(f"<b>{response}</b>"))

# ### Using RssReader

from llama_index import SummaryIndex, RssReader

documents = RssReader().load_data(
    ["https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"]
)

index = SummaryIndex.from_documents(documents)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What happened in the news today?")

