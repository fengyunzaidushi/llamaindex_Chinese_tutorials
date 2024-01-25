#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/node_postprocessor/SentenceTransformerRerank.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Rerank can speed up an LLM query without sacrificing accuracy (and in fact, probably improving it). It does so by pruning away irrelevant nodes from the context.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')
#('pip install git+https://github.com/FlagOpen/FlagEmbedding.git')

from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

import os

OPENAI_API_TOKEN = "sk-"
os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

from llama_index import ServiceContext, set_global_service_context
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
ctx = ServiceContext.from_defaults(embed_model="local")
set_global_service_context(ctx)

# build index
index = VectorStoreIndex.from_documents(documents=documents)

from llama_index.postprocessor import FlagEmbeddingReranker

rerank = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=5)

# First, we try with reranking. We time the query to see how long it takes to process the output from the retrieved context.

from time import time

query_engine = index.as_query_engine(
    similarity_top_k=10, node_postprocessors=[rerank]
)

now = time()
response = query_engine.query(
    "Which grad schools did the author apply for and why?",
)
print(f"Elapsed: {round(time() - now, 2)}s")

print(response)

print(response.get_formatted_sources(length=200))

# Next, we try without rerank

query_engine = index.as_query_engine(similarity_top_k=10)

now = time()
response = query_engine.query(
    "Which grad schools did the author apply for and why?",
)

print(f"Elapsed: {round(time() - now, 2)}s")

print(response)

print(response.get_formatted_sources(length=200))

# As we can see, the query engine with reranking produced a much more concise output in much lower time (6s v.s. 10s). While both responses were essentially correct, the query engine without reranking included a lot of irrelevant information - a phenomenon we could attribute to "pollution of the context window".
