#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/response_synthesizers/tree_summarize.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Tree Summarize

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ## Load Data

from llama_index import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_files=["./data/paul_graham/paul_graham_essay.txt"]
)

docs = reader.load_data()

text = docs[0].text

# ## Summarize

from llama_index.response_synthesizers import TreeSummarize

summarizer = TreeSummarize(verbose=True)

response = await summarizer.aget_response("who is Paul Graham?", [text])

print(response)

