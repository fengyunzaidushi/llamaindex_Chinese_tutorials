#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/PineconeIndexDemo-0.6.0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # LlamaIndex + Pinecone 

# * While Pinecone provides a powerful and efficient retrieval engine,
# it remains challenging to answer complex questions that require multi-step reasoning and synthesis over many data sources.
# * With LlamaIndex, we combine the power of vector similiarty search and multi-step reasoning to delivery higher quality and richer responses.
# 
# 
# Here, we show 2 specific use-cases:
# 1. compare and contrast queries over Wikipedia articles about different cities.
# 2. temporal queries that require reasoning over time

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# #### Creating a Pinecone Index

import pinecone

pinecone.init(environment="eu-west1-gcp")

# create index if it does not already exist
# dimensions are for text-embedding-ada-002
pinecone.create_index(
    "quickstart-index", dimension=1536, metric="euclidean", pod_type="p1"
)

pinecone_index = pinecone.Index("quickstart-index")

# # Use-Case 1: Compare and Contrast

# #### Load Dataset
# 
# Fetch and load Wikipedia pages

from llama_index import SimpleDirectoryReader

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

# #### Build Indices

from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores import PineconeVectorStore

# Build index for each city document
city_indices = {}
index_summaries = {}
for wiki_title in wiki_titles:
    print(f"Building index for {wiki_title}")
    # create storage context
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, namespace=wiki_title
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # build index
    city_indices[wiki_title] = VectorStoreIndex.from_documents(
        city_docs[wiki_title], storage_context=storage_context
    )

    # set summary text for city
    index_summaries[wiki_title] = f"Wikipedia articles about {wiki_title}"

# #### Build Graph Query Engine for Compare & Contrast Query

from llama_index.indices.composability import ComposableGraph
from llama_index.indices.keyword_table.simple_base import (
    SimpleKeywordTableIndex,
)

graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [index for _, index in city_indices.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50,
)

from llama_index.indices.query.query_transform.base import (
    DecomposeQueryTransform,
)
from llama_index.query_engine.transform_query_engine import (
    TransformQueryEngine,
)

decompose_transform = DecomposeQueryTransform(verbose=True)

custom_query_engines = {}
for wiki_title in wiki_titles:
    index = city_indices[wiki_title]
    query_engine = index.as_query_engine()
    query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_extra_info={"index_summary": index_summaries[wiki_title]},
    )
    custom_query_engines[index.index_id] = query_engine

custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
    retriever_mode="simple",
    response_mode="tree_summarize",
)

# with query decomposition in subindices
query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)

# #### Run Compare & Contrast Query

response = query_engine.query(
    "Compare and contrast the demographics in Seattle, Houston, and Toronto."
)

from llama_index.response.pprint_utils import pprint_response

pprint_response(response)

# # Use-Case 2: Temporal Query

# Temporal queries such as "what happened after X" is intuitive to humans, but can often confuse vector databases.  
# 
# This is because the vector embedding will focus on the subject "X" rather than the imporant temporal cue. This results in irrelevant and misleading context that harms the final answer.  
# 
# LlamaIndex solves this by explicitly maintainging node relationships and leverage LLM to automatically perform query expansion to find more relevant context.  

from llama_index import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore

# load documents
documents = SimpleDirectoryReader("../data/paul_graham").load_data()

# define storage context
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, namespace="pg_essay_0.6.0"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# build index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    # override to store Node in document store in addition to vector store, necessary for the node postprocessor
    store_nodes_override=True,
)

# We can define an auto prev/next node postprocessor to leverage LLM reasoning to help query expansion (with relevant additional nodes)

from llama_index.postprocessor.node import (
    AutoPrevNextNodePostprocessor,
)

# define postprocessor
node_postprocessor = AutoPrevNextNodePostprocessor(
    docstore=index.storage_context.docstore,
    service_context=index.service_context,
    num_nodes=3,
    verbose=True,
)

# define query engine
query_engine = index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[node_postprocessor],
)

# #### Example 1

response = query_engine.query(
    "What did the author do after handing off Y Combinator to Sam Altman?",
)

from llama_index.response.pprint_utils import pprint_response

pprint_response(response)

# define query engine
naive_query_engine = index.as_query_engine(
    similarity_top_k=1,
)

response = naive_query_engine.query(
    "What did the author do after handing off Y Combinator to Sam Altman?",
)

pprint_response(response, show_source=True)

# #### Example 2

response = query_engine.query(
    "What did the author do before handing off Y Combinator to Sam Altman?",
)

pprint_response(response, show_source=True)

