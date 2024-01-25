#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/data_connectors/GithubRepositoryReaderDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Github Repo Reader

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# This is due to the fact that we use asyncio.loop_until_complete in
# the DiscordReader. Since the Jupyter kernel itself runs on
# an event loop, we need to add some help with nesting
#('pip install nest_asyncio httpx')
import nest_asyncio

nest_asyncio.apply()

get_ipython().run_line_magic('env', 'OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
from llama_index import VectorStoreIndex, GithubRepositoryReader
from IPython.#display import Markdown, #display
import os

get_ipython().run_line_magic('env', 'GITHUB_TOKEN=github_pat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
github_token = os.environ.get("GITHUB_TOKEN")
owner = "jerryjliu"
repo = "llama_index"
branch = "main"

documents = GithubRepositoryReader(
    github_token=github_token,
    owner=owner,
    repo=repo,
    use_parser=False,
    verbose=False,
    ignore_directories=["examples"],
).load_data(branch=branch)

index = VectorStoreIndex.from_documents(documents)

# import time
# for document in documents:
#     print(document.metadata)
#     time.sleep(.25)
query_engine = index.as_query_engine()
response = query_engine.query(
    "What is the difference between VectorStoreIndex and SummaryIndex?",
    verbose=True,
)

#display(Markdown(f"<b>{response}</b>"))

