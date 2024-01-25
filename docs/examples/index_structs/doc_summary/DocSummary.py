#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/index_structs/doc_summary/DocSummary.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Document Summary Index
# 
# This demo showcases the document summary index, over Wikipedia articles on different cities.
# 
# The document summary index will extract a summary from each document and store that summary, as well as all nodes corresponding to the document.
# 
# Retrieval can be performed through the LLM or embeddings (which is a TODO). We first select the relevant documents to the query based on their summaries. All retrieved nodes corresponding to the selected documents are retrieved.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

#('pip show openai')

import os
import openai

# os.environ["OPENAI_API_KEY"]="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"
# openai.api_key = os.environ["OPENAI_API_KEY"]

api_key1=os.environ['openai_api_key1']

api_key1

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# # Uncomment if you want to temporarily disable logger
# logger = logging.getLogger()
# logger.disabled = True

import nest_asyncio

nest_asyncio.apply()

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI

# ### Load Datasets
# 
# Load Wikipedia pages on different cities

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
city_docs = []
for wiki_title in wiki_titles[:1]:
    docs = SimpleDirectoryReader(
        input_files=[f"data-zh/{wiki_title}.txt"]
    ).load_data()
    docs[0].doc_id = wiki_title
    city_docs.extend(docs)

city_docs

# ### Build Document Summary Index
# 
# We show two ways of building the index:
# - default mode of building the document summary index
# - customizing the summary query
# 

api_base1="https://aigc789.top/v1"

api_base2="https://api.aigc369.com/v1"

# LLM (gpt-3.5-turbo)
chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo",api_base=api_base1,api_key=api_key1)
service_context = ServiceContext.from_defaults(llm=chatgpt, chunk_size=1024)

# default mode of building the index
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)
doc_summary_index = DocumentSummaryIndex.from_documents(
    city_docs,
    service_context=service_context,
    response_synthesizer=response_synthesizer,
    show_progress=True,
)

doc_summary_index.get_document_summary("Toronto")

doc_summary_index.storage_context.persist("index-zh")

from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="index-zh")
doc_summary_index = load_index_from_storage(storage_context)

# ### Perform Retrieval from Document Summary Index
# 
# We show how to execute queries at a high-level. We also show how to perform retrieval at a lower-level so that you can view the parameters that are in place. We show both LLM-based retrieval and embedding-based retrieval using the document summaries.

# #### High-level Querying
# 
# Note: this uses the default, embedding-based form of retrieval

query_engine = doc_summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)

response = query_engine.query("What are the sports teams in Toronto?")

print(response)

# #### LLM-based Retrieval

from llama_index.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
)

retriever = DocumentSummaryIndexLLMRetriever(
    doc_summary_index,
    # choice_select_prompt=None,
    # choice_batch_size=10,
    # choice_top_k=1,
    # format_node_batch_fn=None,
    # parse_choice_select_answer_fn=None,
    # service_context=None
)

retrieved_nodes = retriever.retrieve("What are the sports teams in Toronto?")

print(len(retrieved_nodes))

print(retrieved_nodes[0].score)
print(retrieved_nodes[0].node.get_text())

# use retriever as part of a query engine
from llama_index.query_engine import RetrieverQueryEngine

# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# query
response = query_engine.query("What are the sports teams in Toronto?")
print(response)

# #### Embedding-based Retrieval

from llama_index.indices.document_summary import (
    DocumentSummaryIndexEmbeddingRetriever,
)

retriever = DocumentSummaryIndexEmbeddingRetriever(
    doc_summary_index,
    # similarity_top_k=1,
)

retrieved_nodes = retriever.retrieve("What are the sports teams in Toronto?")

len(retrieved_nodes)

print(retrieved_nodes[0].node.get_text())

# use retriever as part of a query engine
from llama_index.query_engine import RetrieverQueryEngine

# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# query
response = query_engine.query("What are the sports teams in Toronto?")
print(response)

