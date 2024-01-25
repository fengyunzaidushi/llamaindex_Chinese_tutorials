#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/customization/llms/SimpleIndexDemo-ChatGPT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # ChatGPT

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.llms import OpenAI
from IPython.#display import Markdown, #display

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Load documents, build the VectorStoreIndex

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# setup service context
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# #### Query Index

# By default, with the help of langchain's PromptSelector abstraction, we use 
# a modified refine prompt tailored for ChatGPT-use if the ChatGPT model is used.

query_engine = index.as_query_engine(
    service_context=service_context,
    similarity_top_k=3,
    streaming=True,
)
response = query_engine.query(
    "What did the author do growing up?",
)

response.print_response_stream()

query_engine = index.as_query_engine(
    service_context=service_context,
    similarity_top_k=5,
    streaming=True,
)
response = query_engine.query(
    "What did the author do during his time at RISD?",
)

response.print_response_stream()

# **Refine Prompt**: Here is the chat refine prompt 

from llama_index.prompts.chat_prompts import CHAT_REFINE_PROMPT

dict(CHAT_REFINE_PROMPT.prompt)

# #### Query Index (Using the standard Refine Prompt)
# 
# If we use the "standard" refine prompt (where the prompt is one text template instead of multiple messages), we find that the results over ChatGPT are worse. 

from llama_index.prompts.default_prompts import DEFAULT_REFINE_PROMPT

query_engine = index.as_query_engine(
    service_context=service_context,
    refine_template=DEFAULT_REFINE_PROMPT,
    similarity_top_k=5,
    streaming=True,
)
response = query_engine.query(
    "What did the author do during his time at RISD?",
)

response.print_response_stream()

