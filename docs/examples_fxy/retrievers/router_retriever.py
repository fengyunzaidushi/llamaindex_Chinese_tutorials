#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/retrievers/router_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Router Retriever

# 
# The router (`BaseSelector`) module uses the LLM to dynamically make decisions on which underlying retrieval tools to use. This can be helpful to select one out of a diverse range of data sources. This can also be helpful to aggregate retrieval results across a variety of data sources (if a multi-selector module is used).
# 
# This notebook is very similar to the RouterQueryEngine notebook.

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
    SimpleKeywordTableIndex,
)
from llama_index.llms import OpenAI

# ### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Load Data
# 
# We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# initialize service context (set chunk size)
llm = OpenAI(model="gpt-4")
service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm)
nodes = service_context.node_parser.get_nodes_from_documents(documents)

# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# define
summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

list_retriever = summary_index.as_retriever()
vector_retriever = vector_index.as_retriever()
keyword_retriever = keyword_index.as_retriever()

from llama_index.tools import RetrieverTool

list_tool = RetrieverTool.from_defaults(
    retriever=list_retriever,
    description=(
        "Will retrieve all context from Paul Graham's essay on What I Worked"
        " On. Don't use if the question only requires more specific context."
    ),
)
vector_tool = RetrieverTool.from_defaults(
    retriever=vector_retriever,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On."
    ),
)
keyword_tool = RetrieverTool.from_defaults(
    retriever=keyword_retriever,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On (using entities mentioned in query)"
    ),
)

# ### Define Selector Module for Routing
# 
# There are several selectors available, each with some distinct attributes.
# 
# The LLM selectors use the LLM to output a JSON that is parsed, and the corresponding indexes are queried.
# 
# The Pydantic selectors (currently only supported by `gpt-4-0613` and `gpt-3.5-turbo-0613` (the default)) use the OpenAI Function Call API to produce pydantic selection objects, rather than parsing raw JSON.
# 
# Here we use PydanticSingleSelector/PydanticMultiSelector but you can use the LLM-equivalents as well. 

from llama_index.selectors.llm_selectors import (
    LLMSingleSelector,
    LLMMultiSelector,
)
from llama_index.selectors.pydantic_selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.retrievers import RouterRetriever
from llama_index.response.notebook_utils import #display_source_node

# #### PydanticSingleSelector

retriever = RouterRetriever(
    selector=PydanticSingleSelector.from_defaults(llm=llm),
    retriever_tools=[
        list_tool,
        vector_tool,
    ],
)

# will retrieve all context from the author's life
nodes = retriever.retrieve(
    "Can you give me all the context regarding the author's life?"
)
for node in nodes:
    #display_source_node(node)

nodes = retriever.retrieve("What did Paul Graham do after RISD?")
for node in nodes:
    #display_source_node(node)

# #### PydanticMultiSelector

retriever = RouterRetriever(
    selector=PydanticMultiSelector.from_defaults(llm=llm),
    retriever_tools=[list_tool, vector_tool, keyword_tool],
)

nodes = retriever.retrieve(
    "What were noteable events from the authors time at Interleaf and YC?"
)
for node in nodes:
    #display_source_node(node)

nodes = retriever.retrieve(
    "What were noteable events from the authors time at Interleaf and YC?"
)
for node in nodes:
    #display_source_node(node)

nodes = await retriever.aretrieve(
    "What were noteable events from the authors time at Interleaf and YC?"
)
for node in nodes:
    #display_source_node(node)

