#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/query_engine/pydantic_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Query Engine with Pydantic Outputs
# 
# Every query engine has support for integrated structured responses using the following `response_mode`s in `RetrieverQueryEngine`:
# - `refine`
# - `compact`
# - `tree_summarize`
# - `accumulate` (beta, requires extra parsing to convert to objects)
# - `compact_accumulate` (beta, requires extra parsing to convert to objects)
# 

# 
# Under the hood, every LLM response will be a pydantic object. If that response needs to be refined or summarized, it is converted into a JSON string for the next response. Then, the final response is returned as a pydantic object.
# 
# **NOTE:** This can technically work with any LLM, but non-openai is support is still in development and considered beta.

# ## Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# ### Create our Pydanitc Output Object

from typing import List
from pydantic import BaseModel

class Biography(BaseModel):
    """Data model for a biography."""

    name: str
    best_known_for: List[str]
    extra_info: str

# ## Create the Index + Query Engine (OpenAI)
# 
# When using OpenAI, the function calling API will be leveraged for reliable structured outputs.

from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(llm=llm)

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

query_engine = index.as_query_engine(
    output_cls=Biography, response_mode="compact"
)

response = query_engine.query("Who is Paul Graham?")

print(response.name)
print(response.best_known_for)
print(response.extra_info)

# get the full pydanitc object
print(type(response.response))

# ## Create the Index + Query Engine (Non-OpenAI, Beta)
# 
# When using an LLM that does not support function calling, we rely on the LLM to write the JSON itself, and we parse the JSON into the proper pydantic object.

import os

os.environ["ANTHROPIC_API_KEY"] = "sk-..."

from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import Anthropic

llm = Anthropic(model="claude-instant-1.2", temperature=0.1)
service_context = ServiceContext.from_defaults(llm=llm)

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

query_engine = index.as_query_engine(
    output_cls=Biography, response_mode="tree_summarize"
)

response = query_engine.query("Who is Paul Graham?")

print(response.name)
print(response.best_known_for)
print(response.extra_info)

# get the full pydanitc object
print(type(response.response))

# ## Accumulate Examples (Beta)
# 
# Accumulate with pydantic objects requires some extra parsing. This is still a beta feature, but it's still possible to get accumulate pydantic objects.

from typing import List
from pydantic import BaseModel

class Company(BaseModel):
    """Data model for a companies mentioned."""

    company_name: str
    context_info: str

from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(llm=llm)

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

query_engine = index.as_query_engine(
    output_cls=Company, response_mode="accumulate"
)

response = query_engine.query("What companies are mentioned in the text?")

companies = []

# split by the default separator
for response_str in str(response).split("\n---------------------\n"):
    # remove the prefix --  every response starts like `Response 1: {...}`
    # so, we find the first bracket and remove everything before it
    response_str = response_str[response_str.find("{") :]
    companies.append(Company.parse_raw(response_str))

print(companies)

