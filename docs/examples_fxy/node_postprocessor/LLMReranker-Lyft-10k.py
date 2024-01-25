#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/node_postprocessor/LLMReranker-Lyft-10k.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # LLM Reranker Demonstration (2021 Lyft 10-k)
# 
# This tutorial showcases how to do a two-stage pass for retrieval. Use embedding-based retrieval with a high top-k value
# in order to maximize recall and get a large set of candidate items. Then, use LLM-based retrieval
# to dynamically select the nodes that are actually relevant to the query.

import nest_asyncio

nest_asyncio.apply()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.postprocessor import LLMRerank

from llama_index.llms import OpenAI
from IPython.#display import Markdown, #display

# ## Download Data

#("mkdir -p 'data/10k/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'")

# ## Load Data, Build Index

# LLM Predictor (gpt-3.5-turbo) + service context
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")

chunk_overlap = 0
chunk_size = 128

service_context = ServiceContext.from_defaults(
    llm=llm,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

# load documents
documents = SimpleDirectoryReader(
    input_files=["./data/10k/lyft_2021.pdf"]
).load_data()

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# ## Retrieval Comparisons

from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import QueryBundle
import pandas as pd
from IPython.#display import #display, HTML
from copy import deepcopy

pd.set_option("#display.max_colwidth", -1)

def get_retrieved_nodes(
    query_str, vector_top_k=10, reranker_top_n=3, with_reranker=False
):
    query_bundle = QueryBundle(query_str)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)

    if with_reranker:
        # configure reranker
        reranker = LLMRerank(
            choice_batch_size=5,
            top_n=reranker_top_n,
            service_context=service_context,
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

    return retrieved_nodes

def pretty_print(df):
    return #display(HTML(df.to_html().replace("\\n", "<br>")))

def visualize_retrieved_nodes(nodes) -> None:
    result_dicts = []
    for node in nodes:
        node = deepcopy(node)
        node.node.metadata = None
        node_text = node.node.get_text()
        node_text = node_text.replace("\n", " ")

        result_dict = {"Score": node.score, "Text": node_text}
        result_dicts.append(result_dict)

    pretty_print(pd.DataFrame(result_dicts))

new_nodes = get_retrieved_nodes(
    "What is Lyft's response to COVID-19?", vector_top_k=5, with_reranker=False
)

visualize_retrieved_nodes(new_nodes)

new_nodes = get_retrieved_nodes(
    "What is Lyft's response to COVID-19?",
    vector_top_k=20,
    reranker_top_n=5,
    with_reranker=True,
)

visualize_retrieved_nodes(new_nodes)

new_nodes = get_retrieved_nodes(
    "What initiatives are the company focusing on independently of COVID-19?",
    vector_top_k=5,
    with_reranker=False,
)

visualize_retrieved_nodes(new_nodes)

new_nodes = get_retrieved_nodes(
    "What initiatives are the company focusing on independently of COVID-19?",
    vector_top_k=40,
    reranker_top_n=5,
    with_reranker=True,
)

visualize_retrieved_nodes(new_nodes)

