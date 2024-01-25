#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/usecases/10k_sub_question.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # 10K Analysis

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import nest_asyncio

nest_asyncio.apply()

from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI

from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine

# ## Configure LLM service

llm = OpenAI(temperature=0, model="text-davinci-003", max_tokens=-1)
service_context = ServiceContext.from_defaults(llm=llm)

# ## Download Data

#("mkdir -p 'data/10k/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'")

# ## Load data 

lyft_docs = SimpleDirectoryReader(
    input_files=["./data/10k/lyft_2021.pdf"]
).load_data()
uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()

# ## Build indices

lyft_index = VectorStoreIndex.from_documents(lyft_docs)

uber_index = VectorStoreIndex.from_documents(uber_docs)

# ## Build query engines

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)

uber_engine = uber_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021"
            ),
        ),
    ),
]

s_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)

# ## Run queries

response = s_engine.query(
    "Compare and contrast the customer segments and geographies that grew the"
    " fastest"
)

print(response)

response = s_engine.query(
    "Compare revenue growth of Uber and Lyft from 2020 to 2021"
)

print(response)

