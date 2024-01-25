#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/SimpleIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Simple Vector Store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

# #### Load documents, build the VectorStoreIndex

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
)
from IPython.#display import Markdown, #display

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

index = VectorStoreIndex.from_documents(documents)

# save index to disk
index.set_index_id("vector_index")
index.storage_context.persist("./storage")

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="storage")
# load index
index = load_index_from_storage(storage_context, index_id="vector_index")

# #### Query Index

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("What did the author do growing up?")

#display(Markdown(f"<b>{response}</b>"))

# **Query Index with SVM/Linear Regression**
# 
# Use Karpathy's [SVM-based](https://twitter.com/karpathy/status/1647025230546886658?s=20) approach. Set query as positive example, all other datapoints as negative examples, and then fit a hyperplane.

query_modes = [
    "svm",
    "linear_regression",
    "logistic_regression",
]
for query_mode in query_modes:
    # set Logging to DEBUG for more detailed outputs
    query_engine = index.as_query_engine(vector_store_query_mode=query_mode)
    response = query_engine.query("What did the author do growing up?")
    print(f"Query mode: {query_mode}")
    #display(Markdown(f"<b>{response}</b>"))

#display(Markdown(f"<b>{response}</b>"))

print(response.source_nodes[0].text)

# **Query Index with custom embedding string**

from llama_index.schema import QueryBundle

query_bundle = QueryBundle(
    query_str="What did the author do growing up?",
    custom_embedding_strs=["The author grew up painting."],
)
query_engine = index.as_query_engine()
response = query_engine.query(query_bundle)

#display(Markdown(f"<b>{response}</b>"))

# **Use maximum marginal relevance**
# 

query_engine = index.as_query_engine(
    vector_store_query_mode="mmr", vector_store_kwargs={"mmr_threshold": 0.2}
)
response = query_engine.query("What did the author do growing up?")

# #### Get Sources

print(response.get_formatted_sources())

# #### Query Index with Filters
# 
# We can also filter our queries using metadata

from llama_index import Document

doc = Document(text="target", metadata={"tag": "target"})

index.insert(doc)

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="tag", value="target")]
)

retriever = index.as_retriever(
    similarity_top_k=20,
    filters=filters,
)

source_nodes = retriever.retrieve("What did the author do growing up?")

# retrieves only our target node, even though we set the top k to 20
print(len(source_nodes))

print(source_nodes[0].text)
print(source_nodes[0].metadata)

