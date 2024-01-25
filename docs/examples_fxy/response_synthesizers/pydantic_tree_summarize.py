#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/response_synthesizers/pydantic_tree_summarize.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 

# # Pydantic Tree Summarize
# 

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

# # Download Data

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
from llama_index.types import BaseModel
from typing import List

# ### Create pydantic model to structure response

class Biography(BaseModel):
    """Data model for a biography."""

    name: str
    best_known_for: List[str]
    extra_info: str

summarizer = TreeSummarize(verbose=True, output_cls=Biography)

response = summarizer.get_response("who is Paul Graham?", [text])

# ##
# 
# Here, we see the response is in an instance of our `Biography` class.

print(response)

print(response.name)

print(response.best_known_for)

print(response.extra_info)

