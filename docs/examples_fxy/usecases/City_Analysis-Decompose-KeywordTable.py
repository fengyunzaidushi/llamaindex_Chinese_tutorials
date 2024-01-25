#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/usecases/City_Analysis-Decompose-KeywordTable.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Test Complex Queries over Multiple Documents (with and without Query Decomposition)
# 
# Query Decomposition: The ability to decompose a complex query into a simpler query given the content of the index.
# 
# Use OpenAI as the LLM model and embedding model.

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
    SummaryIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
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

data_path = Path("data_wiki")

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

    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

# Load all wiki documents
city_docs = {}
all_docs = []
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[data_path / f"{wiki_title}.txt"]
    ).load_data()
    all_docs.extend(city_docs[wiki_title])

# define service context
service_context = ServiceContext.from_defaults(
    chunk_size=512,
)

# ### Building the document indices
# Build a separate vector index for each wiki pages about cities.
# 
# We also build a "global" vector index, which ingest documents for *all* cities. 
# 
# This allows us to test different types of data structures!

# Build index for each city document
city_indices = {}
index_summaries = {}
for wiki_title in wiki_titles:
    print(f"Building index for {wiki_title}")
    city_indices[wiki_title] = VectorStoreIndex.from_documents(
        city_docs[wiki_title], service_context=service_context
    )
    # set summary text for city
    index_summaries[wiki_title] = f"Wikipedia articles about {wiki_title}"

# also setup a global vector index
global_index = VectorStoreIndex.from_documents(
    all_docs, service_context=service_context
)

# ### Creating the right structure to run compare/contrast queries
# 
# Our key goal in this notebook is to run compare/contrast queries between different cities.
# 
# We currently have a separate vector index for every city document. We want to setup a "graph" structure in order to route the query 
# in the right manner in order to retrieve the relevant text sections for each city. 
# 
# We compose a keyword table index on top of all the vector indices.

from llama_index.indices.composability import ComposableGraph

graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [index for _, index in city_indices.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50,
)

# ### Define Query Transformation + Query Configs
# 
# We also define a "query decomposition" transform. Since we have a graph structure over multiple indexes, query decomposition
# allows us to break a complex question into a simpler one over a given index.
# 
# This works well in comparing/contrasting different cities because it allows us to ask questions specific to each city.

# **Query Transform**

from llama_index.indices.query.query_transform.base import (
    DecomposeQueryTransform,
)

decompose_transform = DecomposeQueryTransform(verbose=True)

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from llama_index.query_engine.transform_query_engine import (
    TransformQueryEngine,
)

custom_query_engines = {}
for index in city_indices.values():
    query_engine = index.as_query_engine(service_context=service_context)
    query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_extra_info={"index_summary": index.index_struct.summary},
    )
    custom_query_engines[index.index_id] = query_engine
custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
    retriever_mode="simple",
    response_mode="tree_summarize",
    service_context=service_context,
)

# ### Let's Run Some Queries! 
# 
# We run queries over the graphs and analyze the results.
# 
# We also compare results against the baseline global vector index. In the majority of cases the global vector index provides insufficient answers.

# **Complex Query 1**

# with query decomposition in subindices
query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)
query_str = (
    "Compare and contrast the demographics in Seattle, Houston, and Toronto. "
)

response = query_engine.query(query_str)

print(str(response))

query_engine = global_index.as_query_engine(
    similarity_top_k=3, response_mode="tree_summarize"
)
response = query_engine.query(query_str)

# NOTE: the global vector index seems to provide the right results....
# BUT see below!
print(str(response))

# NOTE: there's hallucination! the sources only reference Toronto
print(response.source_nodes[0].source_text)
print(response.source_nodes[1].source_text)

# **Complex Query 2**

# with query decomposition
query_str = "What are the basketball teams in Houston and Boston?"

query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)

response = query_engine.query(query_str)

print(str(response))

query_engine = global_index.as_query_engine(
    similarity_top_k=2, response_mode="tree_summarize"
)
response = query_engine.query(query_str)

print(str(response))

# **Complex Query 3**

# with query decomposition
query_str = "Compare and contrast the climate of Houston and Boston "

query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)

response = query_engine.query(query_str)

print(response)

query_engine = global_index.as_query_engine(
    similarity_top_k=2, response_mode="tree_summarize"
)
response = query_engine.query(query_str)

print(str(response))

