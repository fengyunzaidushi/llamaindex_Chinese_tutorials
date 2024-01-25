#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/SimpleIndexDemoLlama2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Llama2 + VectorStoreIndex
# 
# This notebook walks through the proper setup to use llama-2 with LlamaIndex. Specifically, we look at using a vector store index.

# ## Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ### Keys

import os

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["REPLICATE_API_TOKEN"] = "REPLICATE_API_TOKEN"

# currently needed for notebooks
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# ### Load documents, build the VectorStoreIndex

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)

from IPython.#display import Markdown, #display

from llama_index.llms import Replicate
from llama_index import ServiceContext, set_global_service_context
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

# The replicate endpoint
LLAMA_13B_V2_CHAT = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"

# inject custom system prompt into llama-2
def custom_completion_to_prompt(completion: str) -> str:
    return completion_to_prompt(
        completion,
        system_prompt=(
            "You are a Q&A assistant. Your goal is to answer questions as "
            "accurately as possible is the instructions and context provided."
        ),
    )

llm = Replicate(
    model=LLAMA_13B_V2_CHAT,
    temperature=0.01,
    # override max tokens since it's interpreted
    # as context window instead of max tokens
    context_window=4096,
    # override completion representation for llama 2
    completion_to_prompt=custom_completion_to_prompt,
    # if using llama 2 for data agents, also override the message representation
    messages_to_prompt=messages_to_prompt,
)

# set a global service context
ctx = ServiceContext.from_defaults(llm=llm)
set_global_service_context(ctx)

# Download Data

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

index = VectorStoreIndex.from_documents(documents)

# ## Querying

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")
#display(Markdown(f"<b>{response}</b>"))

# ### Streaming Support

query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("What happened at interleaf?")
for token in response.response_gen:
    print(token, end="")

