#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/agent/Chatbot_SEC.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # 💬🤖 How to Build a Chatbot
# 
# LlamaIndex serves as a bridge between your data and Language Learning Models (LLMs), providing a toolkit that enables you to establish a query interface around your data for a variety of tasks, such as question-answering and summarization.
# 

# 
# **Note**: This tutorial builds upon initial work on creating a query interface over SEC 10-K filings - [check it out here](https://medium.com/@jerryjliu98/how-unstructured-and-llamaindex-can-help-bring-the-power-of-llms-to-your-own-data-3657d063e30d).
# 
# ### Context
# 

# ### Preparation

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

import nest_asyncio

nest_asyncio.apply()

# set text wrapping
from IPython.#display import HTML, #display

def set_css():
    #display(
        HTML(
            """
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  """
        )
    )

get_ipython().events.register("pre_run_cell", set_css)

# ##
# 
# Let's first download the raw 10-k files, from 2019-2022.

# NOTE: the code examples assume you're operating within a Jupyter notebook.
# download files
#('mkdir data')
#('wget "https://www.dropbox.com/s/948jr9cfs7fgj99/UBER.zip?dl=1" -O data/UBER.zip')
#('unzip data/UBER.zip -d data')

# To parse the HTML files into formatted text, we use the [Unstructured](https://github.com/Unstructured-IO/unstructured) library. Thanks to [LlamaHub](https://llamahub.ai/), we can directly integrate with Unstructured, allowing conversion of any text into a Document format that LlamaIndex can ingest.
# 
# First we install the necessary packages:

#('pip install llama-hub unstructured')

# Then we can use the `UnstructuredReader` to parse the HTML files into a list of `Document` objects.

from llama_hub.file.unstructured.base import UnstructuredReader
from pathlib import Path

years = [2022, 2021, 2020, 2019]

loader = UnstructuredReader()
doc_set = {}
all_docs = []
for year in years:
    year_docs = loader.load_data(
        file=Path(f"./data/UBER/UBER_{year}.html"), split_documents=False
    )
    # insert year metadata into each year
    for d in year_docs:
        d.metadata = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)

# ### Setting up Vector Indices for each year
# 
# We first setup a vector index for each year. Each vector index allows us
# to ask questions about the 10-K filing of a given year.
# 
# We build each index and save it to disk.

# initialize simple vector indices
# NOTE: don't run this cell if the indices are already loaded!
from llama_index import VectorStoreIndex, ServiceContext, StorageContext

index_set = {}
service_context = ServiceContext.from_defaults(chunk_size=512)
for year in years:
    storage_context = StorageContext.from_defaults()
    cur_index = VectorStoreIndex.from_documents(
        doc_set[year],
        service_context=service_context,
        storage_context=storage_context,
    )
    index_set[year] = cur_index
    storage_context.persist(persist_dir=f"./storage/{year}")

# To load an index from disk, do the following

# Load indices from disk
from llama_index import load_index_from_storage

index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults(
        persist_dir=f"./storage/{year}"
    )
    cur_index = load_index_from_storage(
        storage_context, service_context=service_context
    )
    index_set[year] = cur_index

# ### Setting up a Sub Question Query Engine to Synthesize Answers Across 10-K Filings
# 
# Since we have access to documents of 4 years, we may not only want to ask questions regarding the 10-K document of a given year, but ask questions that require analysis over all 10-K filings.
# 
# To address this, we can use a [Sub Question Query Engine](https://gpt-index.readthedocs.io/en/stable/examples/query_engine/sub_question_query_engine.html). It decomposes a query into subqueries, each answered by an individual vector index, and synthesizes the results to answer the overall query.
# 
# LlamaIndex provides some wrappers around indices (and query engines) so that they can be used by query engines and agents. First we define a `QueryEngineTool` for each vector index.
# Each tool has a name and a description; these are what the LLM agent sees to decide which tool to choose.

from llama_index.tools import QueryEngineTool, ToolMetadata

individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index_set[year].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_{year}",
            description=(
                "useful for when you want to answer queries about the"
                f" {year} SEC 10-K for Uber"
            ),
        ),
    )
    for year in years
]

# Now we can create the Sub Question Query Engine, which will allow us to synthesize answers across the 10-K filings. We pass in the `individual_query_engine_tools` we defined above, as well as a `service_context` that will be used to run the subqueries.

from llama_index.query_engine import SubQuestionQueryEngine

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
    service_context=service_context,
)

# ### Setting up the Chatbot Agent
# 
# We use a LlamaIndex Data Agent to setup the outer chatbot agent, which has access to a set of Tools. Specifically, we will use an OpenAIAgent, that takes advantage of OpenAI API function calling. We want to use the separate Tools we defined previously for each index (corresponding to a given year), as well as a tool for the sub question query engine we defined above.
# 
# First we define a `QueryEngineTool` for the sub question query engine:

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description=(
            "useful for when you want to answer queries that require analyzing"
            " multiple SEC 10-K documents for Uber"
        ),
    ),
)

# Then, we combine the Tools we defined above into a single list of tools for the agent:

tools = individual_query_engine_tools + [query_engine_tool]

# Finally, we call `OpenAIAgent.from_tools` to create the agent, passing in the list of tools we defined above.

from llama_index.agent import OpenAIAgent

agent = OpenAIAgent.from_tools(tools, verbose=True)

# ### Testing the Agent
# 
# We can now test the agent with various queries.
# 
# If we test it with a simple "hello" query, the agent does not use any Tools.

response = agent.chat("hi, i am bob")
print(str(response))

# If we test it with a query regarding the 10-k of a given year, the agent will use
# the relevant vector index Tool.

response = agent.chat(
    "What were some of the biggest risk factors in 2020 for Uber?"
)
print(str(response))

# Finally, if we test it with a query to compare/contrast risk factors across years, the agent will use the Sub Question Query Engine Tool.

cross_query_str = (
    "Compare/contrast the risk factors described in the Uber 10-K across"
    " years. Give answer in bullet points."
)

response = agent.chat(cross_query_str)
print(str(response))

# ### Setting up the Chatbot Loop
# 
# Now that we have the chatbot setup, it only takes a few more steps to setup a basic interactive loop to chat with our SEC-augmented chatbot!

agent = OpenAIAgent.from_tools(tools)  # verbose=False by default

while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = agent.chat(text_input)
    print(f"Agent: {response}")

# User: What were some of the legal proceedings against Uber in 2022?

