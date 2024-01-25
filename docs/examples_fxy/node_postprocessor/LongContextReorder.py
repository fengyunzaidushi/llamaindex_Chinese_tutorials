#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/node_postprocessor/LongContextReorder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # LongContextReorder
# 
# Models struggle to access significant details found in the center of extended contexts. [A study](https://arxiv.org/abs/2307.03172) observed that the best performance typically arises when crucial data is positioned at the start or conclusion of the input context. Additionally, as the input context lengthens, performance drops notably, even in models designed for long contexts.
# 
# This module will re-order the retrieved nodes, which can be helpful in cases where a large top-k is needed.

# ## Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index import ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.1)
ctx = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-base-en-v1.5"
)

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents, service_context=ctx)

# ## Run Query

from llama_index.postprocessor import LongContextReorder

reorder = LongContextReorder()

reorder_engine = index.as_query_engine(
    node_postprocessors=[reorder], similarity_top_k=5
)
base_engine = index.as_query_engine(similarity_top_k=5)

from llama_index.response.notebook_utils import #display_response

base_response = base_engine.query("Did the author meet Sam Altman?")
#display_response(base_response)

reorder_response = reorder_engine.query("Did the author meet Sam Altman?")
#display_response(reorder_response)

# #

print(base_response.get_formatted_sources())

print(reorder_response.get_formatted_sources())

