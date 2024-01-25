#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/composable_indices/city_analysis/PineconeDemo-CityAnalysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Using LlamaIndex with Pinecone
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
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.vector_stores import PineconeVectorStore
from llama_index.llms import OpenAI

# #### Load Datasets
# 
# Load Wikipedia pages

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
pinecone_titles = [
    "toronto",
    "seattle",
    "san-francisco",
    "chicago",
    "boston",
    "dc",
    "cambridge",
    "houston",
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

# #

import pinecone
import os

api_key = ""
environment = "eu-west1-gcp"
index_name = "quickstart"

os.environ["PINECONE_API_KEY"] = api_key

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

# ### Recommended Option:  Pass API key via env variable, and index_name & environment as argument

# Build city document index
from llama_index.storage.storage_context import StorageContext

city_indices = {}
for pinecone_title, wiki_title in zip(pinecone_titles, wiki_titles):
    metadata_filters = {"wiki_title": wiki_title}
    vector_store = PineconeVectorStore(
        index_name=index_name,
        environment=environment,
        metadata_filters=metadata_filters,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    city_indices[wiki_title] = VectorStoreIndex.from_documents(
        city_docs[wiki_title],
        storage_context=storage_context,
        service_context=service_context,
    )
    # set summary text for city
    city_indices[wiki_title].index_struct.index_id = pinecone_title

# ### Alternative Option: instantiate pinecone client first, then pass to PineconeVectorStore

pinecone.init(api_key=api_key, environment=environment)

pinecone_index = pinecone.Index(index_name)

# Build city document index
city_indices = {}
for pinecone_title, wiki_title in zip(pinecone_titles, wiki_titles):
    metadata_filters = {"wiki_title": wiki_title}
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, metadata_filters=metadata_filters
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    city_indices[wiki_title] = VectorStoreIndex.from_documents(
        city_docs[wiki_title],
        storage_context=storage_context,
        service_context=service_context,
    )
    # set summary text for city
    city_indices[wiki_title].index_struct.index_id = pinecone_title

# ### Query Index

response = (
    city_indices["Boston"]
    .as_query_engine(service_context=service_context)
    .query("Tell me about the arts and culture of Boston")
)

print(str(response))
print(response.get_formatted_sources())

# ### Build Graph: Keyword Table Index on top of vector indices! 
# 
# We compose a keyword table index on top of all the vector indices.

from llama_index.indices.composability.graph import ComposableGraph

# set summaries for each city
index_summaries = {}
for wiki_title in wiki_titles:
    # set summary text for city
    index_summaries[wiki_title] = f"Wikipedia articles about {wiki_title}"

graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [index for _, index in city_indices.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50,
)

custom_query_engines = {
    graph.root_id: graph.root_index.as_query_engine(
        retriever_mode="simple", service_context=service_context
    )
}

query_engine = graph.as_query_engine(
    custom_query_engines=custom_query_engines,
)

# ### Compare Queries (text-davinci-003 vs. ChatGPT)

# **Simple Query**

query_str = "Tell me more about Boston"
response_chatgpt = query_engine.query(query_str)

print(response_chatgpt)
print(response_chatgpt.get_formatted_sources())

