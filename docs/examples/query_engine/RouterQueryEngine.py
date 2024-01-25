#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/query_engine/RouterQueryEngine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Router Query Engine

# ### Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# NOTE: This is ONLY necessary in jupyter notebook.
# Details: Jupyter runs an event-loop behind the scenes.
#          This results in nested event-loops when we start an event-loop to make async queries.
#          This is normally not allowed, we use nest_asyncio to allow it for convenience.
import nest_asyncio

nest_asyncio.apply()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)

# ### Load Data
# 
# We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.

# load documents
documents = SimpleDirectoryReader("../data/paul_graham").load_data()

# initialize service context (set chunk size)
service_context = ServiceContext.from_defaults(chunk_size=1024)
nodes = service_context.node_parser.get_nodes_from_documents(documents)

# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# ### Define Summary Index and Vector Index over Same Data 

summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

# ### Define Query Engines and Set Metadata

list_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()

from llama_index.tools.query_engine import QueryEngineTool

list_tool = QueryEngineTool.from_defaults(
    query_engine=list_query_engine,
    description=(
        "Useful for summarization questions related to Paul Graham eassy on"
        " What I Worked On."
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On."
    ),
)

# ### Define Router Query Engine
# 
# There are several selectors available, each with some distinct attributes.
# 
# The LLM selectors use the LLM to output a JSON that is parsed, and the corresponding indexes are queried.
# 
# The Pydantic selectors (currently only supported by `gpt-4-0613` and `gpt-3.5-turbo-0613` (the default)) use the OpenAI Function Call API to produce pydantic selection objects, rather than parsing raw JSON.
# 
# For each type of selector, there is also the option to select 1 index to route to, or multiple.
# 
# #### PydanticSingleSelector
# 
# Use the OpenAI Function API to generate/parse pydantic objects under the hood for the router selector.

from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.llm_selectors import (
    LLMSingleSelector,
    LLMMultiSelector,
)
from llama_index.selectors.pydantic_selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)

query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)

response = query_engine.query("What is the summary of the document?")
print(str(response))

response = query_engine.query("What did Paul Graham do after RICS?")
print(str(response))

# #### LLMSingleSelector
# 
# Use OpenAI (or any other LLM) to parse generated JSON under the hood to select a sub-index for routing.

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)

response = query_engine.query("What is the summary of the document?")
print(str(response))

response = query_engine.query("What did Paul Graham do after RICS?")
print(str(response))

# [optional] look at selected results
print(str(response.metadata["selector_result"]))

# #### PydanticMultiSelector
# 

from llama_index import SimpleKeywordTableIndex

keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

keyword_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context using keywords from Paul"
        " Graham essay on What I Worked On."
    ),
)

query_engine = RouterQueryEngine(
    selector=PydanticMultiSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
        keyword_tool,
    ],
)

# This query could use either a keyword or vector query engine, so it will combine responses from both
response = query_engine.query(
    "What were noteable events and people from the authors time at Interleaf"
    " and YC?"
)
print(str(response))

# [optional] look at selected results
print(str(response.metadata["selector_result"]))

