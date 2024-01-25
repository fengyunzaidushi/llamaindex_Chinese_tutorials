#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/usecases/10q_sub_question.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # 10Q Analysis

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import nest_asyncio

nest_asyncio.apply()

from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.response.pprint_utils import pprint_response
from llama_index.llms import OpenAI

from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine

# ## Configure LLM service

llm = OpenAI(temperature=0, model="text-davinci-003", max_tokens=-1)
service_context = ServiceContext.from_defaults(llm=llm)

# ## Download Data

#("mkdir -p 'data/10q/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10q/uber_10q_march_2022.pdf' -O 'data/10q/uber_10q_march_2022.pdf'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10q/uber_10q_june_2022.pdf' -O 'data/10q/uber_10q_june_2022.pdf'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10q/uber_10q_sept_2022.pdf' -O 'data/10q/uber_10q_sept_2022.pdf'")

# ## Load data

march_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_march_2022.pdf"]
).load_data()
june_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_june_2022.pdf"]
).load_data()
sept_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_sept_2022.pdf"]
).load_data()

# # Build indices

march_index = VectorStoreIndex.from_documents(march_2022)
june_index = VectorStoreIndex.from_documents(june_2022)
sept_index = VectorStoreIndex.from_documents(sept_2022)

# ## Build query engines

march_engine = march_index.as_query_engine(similarity_top_k=3)
june_engine = june_index.as_query_engine(similarity_top_k=3)
sept_engine = sept_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=sept_engine,
        metadata=ToolMetadata(
            name="sept_22",
            description=(
                "Provides information about Uber quarterly financials ending"
                " September 2022"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=june_engine,
        metadata=ToolMetadata(
            name="june_22",
            description=(
                "Provides information about Uber quarterly financials ending"
                " June 2022"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=march_engine,
        metadata=ToolMetadata(
            name="march_22",
            description=(
                "Provides information about Uber quarterly financials ending"
                " March 2022"
            ),
        ),
    ),
]

s_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)

# ## Run queries

response = s_engine.query(
    "Analyze Uber revenue growth over the latest two quarter filings"
)

print(response)

response = s_engine.query(
    "Analyze change in macro environment over the 3 quarters"
)

print(response)

response = s_engine.query("How much cash did Uber have in sept 2022")

print(response)

