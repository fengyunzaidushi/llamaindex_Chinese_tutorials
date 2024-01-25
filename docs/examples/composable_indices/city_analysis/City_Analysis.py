#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/composable_indices/city_analysis/City_Analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Test Complex Queries over Multiple Documents (text-davinci-003 vs. ChatGPT)
# 
# Test complex queries over both text-davinci-003 and ChatGPT

#('pip install llama-index')

# My OpenAI Key
import os

os.environ["OPENAI_API_KEY"] = ""

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SummaryIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.llms import OpenAI
import requests

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

# LLM Predictor (text-davinci-003)
davinci = OpenAI(temperature=0, model="text-davinci-003")
service_context_davinci = ServiceContext.from_defaults(llm=davinci)

# # LLM Predictor (gpt-3.5-turbo)
chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context_chatgpt = ServiceContext.from_defaults(llm=chatgpt)

# Build city document index
city_indices = {}
for wiki_title in wiki_titles:
    city_indices[wiki_title] = VectorStoreIndex.from_documents(
        city_docs[wiki_title]
    )

# ### Build Graph: Keyword Table Index on top of vector indices! 
# 
# We compose a keyword table index on top of all the vector indices.

# set summaries for each city
index_summaries = {}
for wiki_title in wiki_titles:
    # set summary text for city
    index_summaries[wiki_title] = f"Wikipedia articles about {wiki_title}"

from llama_index.indices.composability import ComposableGraph

graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [index for _, index in city_indices.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50,
)

# ### Compare Queries (text-davinci-003 vs. ChatGPT)

# **Simple Query**

query_engine_davinci = graph.as_query_engine(
    custom_query_engines={
        graph.root_index.index_id: graph.root_index.as_query_engine(
            retriever_mode="simple",
            service_context=service_context_davinci,
            response_mode="tree_summarize",
        )
    }
)
query_engine_chatgpt = graph.as_query_engine(
    custom_query_engines={
        graph.root_index.index_id: graph.root_index.as_query_engine(
            retriever_mode="simple",
            service_context=service_context_chatgpt,
            response_mode="tree_summarize",
        )
    }
)
query_str = "Tell me more about Boston"
response_davinci = query_engine_davinci.query(query_str)
response_chatgpt = query_engine_chatgpt.query(query_str)

print(response_davinci)

print(response_chatgpt)

# **Complex Query 1**

query_str = (
    "Tell me the airports in Seattle, Houston, and Toronto. If only one city"
    " is provided, return the airport information for that city. If airports"
    " for multiple cities are provided, compare and contrast the airports. "
)
response_davinci = query_engine_davinci.query(query_str)
response_chatgpt = query_engine_chatgpt.query(query_str)

print(response_davinci)

print(response_chatgpt)

# **Complex Query 2**

query_str = (
    "Look at Houston and Boston. If only one city is provided, provide"
    " information about the sports teams for that city. If context for"
    " multiple cities are provided, compare and contrast the sports"
    " environment of the cities. "
)
response_davinci = query_engine_davinci.query(query_str)
response_chatgpt = query_engine_chatgpt.query(query_str)

print(response_davinci)

print(response_chatgpt)

# **Complex Query 3**

query_str = (
    "Look at Houston and Boston. If only one city is provided, provide"
    " information about the arts and culture for that city. If context for"
    " multiple cities are provided, compare and contrast the arts and culture"
    " of the two cities. "
)
response_davinci = query_engine_davinci.query(query_str)
response_chatgpt = query_engine_chatgpt.query(query_str)

print(response_davinci)

print(response_chatgpt)

# **Complex Query 4**

query_str = (
    "Look at Toronto and San Francisco. If only one city is provided, provide"
    " information about the demographics for that city. If context for"
    " multiple cities are provided, compare and contrast the demographics of"
    " the two cities. "
)
response_davinci = query_engine_davinci.query(query_str)
response_chatgpt = query_engine_chatgpt.query(query_str)

print(response_davinci)

print(response_chatgpt)

