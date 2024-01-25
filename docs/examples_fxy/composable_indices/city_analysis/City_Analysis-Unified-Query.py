#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/composable_indices/city_analysis/City_Analysis-Unified-Query.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Defining a Unified Query Interface over your Data

# This notebook shows how to build a unified query interface that can handle:
# 1. **heterogeneous data sources** (e.g. data about multiple cities) and 
# 2. **complex queries** (e.g. compare and contrast).

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Uncomment if you want to temporarily disable logger
logger = logging.getLogger()
logger.disabled = True

from llama_index import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
)

# #### Load Datasets
# 
# Load Wikipedia pages about different cities.

wiki_titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]

from pathlib import Path

import requests

for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

# Load all wiki documents
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()

# ### Building Vector Indices
# Build a vector index for the wiki pages about cities.

from llama_index.llms import OpenAI

chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=chatgpt, chunk_size=1024)

gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context = ServiceContext.from_defaults(llm=gpt4, chunk_size=1024)

# Build city document index
vector_indices = {}
for wiki_title in wiki_titles:
    # build vector index
    vector_indices[wiki_title] = VectorStoreIndex.from_documents(
        city_docs[wiki_title], service_context=service_context
    )

    # set id for vector index
    vector_indices[wiki_title].set_index_id(wiki_title)

index_summaries = {
    wiki_title: (
        f"This content contains Wikipedia articles about {wiki_title}. Use"
        " this index if you need to lookup specific facts about"
        f" {wiki_title}.\nDo not use this index if you want to analyze"
        " multiple cities."
    )
    for wiki_title in wiki_titles
}

# #### Test Querying the Vector Index

query_engine = vector_indices["Toronto"].as_query_engine()
response = query_engine.query("What are the sports teams in Toronto?")

print(str(response))

# ### Build a Graph for Compare/Contrast Queries
# 
# We build a graph by composing a keyword table index on top of all the vector indices.
# We use this graph for compare/contrast queries

from llama_index.indices.composability import ComposableGraph

graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [index for _, index in vector_indices.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50,
)

# get root index
root_index = graph.get_index(graph.root_id)

# set id of root index
root_index.set_index_id("compare_contrast")

# define decompose_transform
from llama_index.indices.query.query_transform.base import (
    DecomposeQueryTransform,
)

decompose_transform = DecomposeQueryTransform(llm=chatgpt, verbose=True)

# define custom retrievers
from llama_index.query_engine.transform_query_engine import (
    TransformQueryEngine,
)

custom_query_engines = {}
for index in vector_indices.values():
    query_engine = index.as_query_engine(service_context=service_context)
    query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_metadata={"index_summary": index.index_struct.summary},
    )
    custom_query_engines[index.index_id] = query_engine

custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
    retriever_mode="simple",
    response_mode="tree_summarize",
    service_context=service_context,
    verbose=True,
)

# define graph
graph_query_engine = graph.as_query_engine(
    custom_query_engines=custom_query_engines
)

# #### Test querying the graph

query_str = "Compare and contrast the arts and culture of Houston and Boston. "
response = graph_query_engine.query(query_str)

print(response)

# ### Build a router to automatically choose between indices and graph

# We can use a `RouterQueryEngine` to automatically route to the vector indices and the graph.
# 

# 
# To do this, first build the query engines, and give each a description to obtain a `QueryEngineTool`.

from llama_index.tools.query_engine import QueryEngineTool

query_engine_tools = []

# add vector index tools
for wiki_title in wiki_titles:
    index = vector_indices[wiki_title]
    summary = index_summaries[wiki_title]

    query_engine = index.as_query_engine(service_context=service_context)
    vector_tool = QueryEngineTool.from_defaults(
        query_engine, description=summary
    )
    query_engine_tools.append(vector_tool)

# add graph tool
graph_description = (
    "This tool contains Wikipedia articles about multiple cities. "
    "Use this tool if you want to compare multiple cities. "
)
graph_tool = QueryEngineTool.from_defaults(
    graph_query_engine, description=graph_description
)
query_engine_tools.append(graph_tool)

# Then, define the `RouterQueryEngine` with a desired selector module. 
# Here, we use the `LLMSingleSelector`, which uses LLM to choose a underlying query engine to route the query to.

from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.llm_selectors import LLMSingleSelector

router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(service_context=service_context),
    query_engine_tools=query_engine_tools,
)

# Asking a compare and contrast question should route the query to the graph.

# ask a compare/contrast question
response = router_query_engine.query(
    "Compare and contrast the arts and culture of Houston and Boston.",
)

print(response)

# Asking a question about a specific city should route the query to the specific vector index query engine.

response = router_query_engine.query("What are the sports teams in Toronto?")

print(response)

