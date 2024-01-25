#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/composable_indices/city_analysis/City_Analysis-Decompose.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Test Complex Queries over Multiple Documents (with and without Query Decomposition)
# 
# Query Decomposition: The ability to decompose a complex query into a simpler query given the content of the index.
# 
# Use ChatGPT as the LLM model

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging

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
# Load Wikipedia pages as well as Paul Graham's "What I Worked On" essay

wiki_titles = [
    "Toronto",
    "Seattle",
    "San Francisco",
    "Chicago",
    "Boston",
    "Washington, D.C.",
    "Cambridge, Massachusetts",
    "Houston",
]

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

# ### Building the document indices
# Build a vector index for the wiki pages about cities and persons, and PG essay

# # LLM Predictor (gpt-3.5-turbo)
from llama_index.llms.openai import OpenAI

chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=chatgpt)

# Build city document index
city_indices = {}
index_summaries = {}
for wiki_title in wiki_titles:
    city_indices[wiki_title] = VectorStoreIndex.from_documents(
        city_docs[wiki_title], service_context=service_context
    )
    # set summary text for city
    index_summaries[wiki_title] = f"Wikipedia articles about {wiki_title}"

# ### Build Graph: Keyword Table Index on top of vector indices! 
# 
# We compose a keyword table index on top of all the vector indices.

from llama_index.indices.composability import ComposableGraph

graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [index for _, index in city_indices.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50,
)

# ### Define Query Configs

# **Query Transform**

from llama_index.indices.query.query_transform.base import (
    DecomposeQueryTransform,
)

decompose_transform = DecomposeQueryTransform(
    service_context.llm, verbose=True
)

# **Complex Query 1**

# with query decomposition in subindices
from llama_index.query_engine.transform_query_engine import (
    TransformQueryEngine,
)

custom_query_engines = {}
for index in city_indices.values():
    query_engine = index.as_query_engine(service_context=service_context)
    transform_metadata = {"index_summary": index.index_struct.summary}
    tranformed_query_engine = TransformQueryEngine(
        query_engine,
        decompose_transform,
        transform_metadata=transform_metadata,
    )
    custom_query_engines[index.index_id] = tranformed_query_engine

custom_query_engines[
    graph.root_index.index_id
] = graph.root_index.as_query_engine(
    retriever_mode="simple",
    response_mode="tree_summarize",
    service_context=service_context,
)

query_engine_decompose = graph.as_query_engine(
    custom_query_engines=custom_query_engines,
)

response_chatgpt = query_engine_decompose.query(
    "Compare and contrast the airports in Seattle, Houston, and Toronto. "
)

print(str(response_chatgpt))

# without query decomposition in subindices

custom_query_engines = {}
for index in city_indices.values():
    query_engine = index.as_query_engine(service_context=service_context)
    custom_query_engines[index.index_id] = query_engine

custom_query_engines[
    graph.root_index.index_id
] = graph.root_index.as_query_engine(
    retriever_mode="simple",
    response_mode="tree_summarize",
    service_context=service_context,
)

query_engine = graph.as_query_engine(
    custom_query_engines=custom_query_engines,
)

response_chatgpt = query_engine.query(
    "Compare and contrast the airports in Seattle, Houston, and Toronto. "
)

str(response_chatgpt)

# **Complex Query 2**

# with query decomposition
response_chatgpt = query_engine_decompose.query(
    "Compare and contrast the sports environment of Houston and Boston. "
)

str(response_chatgpt)

# without query decomposition
response_chatgpt = query_engine.query(
    "Compare and contrast the sports environment of Houston and Boston. "
)

str(response_chatgpt)

# with query decomposition
response_chatgpt = query_engine_decompose.query(
    "Compare and contrast the sports environment of Houston and Boston. "
)

print(response_chatgpt)

# without query decomposition
response_chatgpt = query_engine.query(
    "Compare and contrast the sports environment of Houston and Boston. "
)

print(response_chatgpt)

# **Complex Query 3**

# with query decomposition
response_chatgpt = query_engine_decompose.query(
    "Compare and contrast the arts and culture of Houston and Boston. "
)

print(response_chatgpt)

# without query decomposition
response_chatgpt = query_engine.query(
    "Compare and contrast the arts and culture of Houston and Boston. "
)

print(response_chatgpt)

