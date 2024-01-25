#!/usr/bin/env python
# coding: utf-8

# # LLM Reranker Demonstration (Great Gatsby)
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

# ## Load Data, Build Index

# LLM Predictor (gpt-3.5-turbo) + service context
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

# load documents
documents = SimpleDirectoryReader("../../../examples/gatsby/data").load_data()

documents

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# ## Retrieval

from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import QueryBundle
import pandas as pd
from IPython.#display import #display, HTML

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
        result_dict = {"Score": node.score, "Text": node.node.get_text()}
        result_dicts.append(result_dict)

    pretty_print(pd.DataFrame(result_dicts))

new_nodes = get_retrieved_nodes(
    "Who was driving the car that hit Myrtle?",
    vector_top_k=3,
    with_reranker=False,
)

visualize_retrieved_nodes(new_nodes)

new_nodes = get_retrieved_nodes(
    "Who was driving the car that hit Myrtle?",
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=True,
)

visualize_retrieved_nodes(new_nodes)

new_nodes = get_retrieved_nodes(
    "What did Gatsby want Daisy to do in front of Tom?",
    vector_top_k=3,
    with_reranker=False,
)

visualize_retrieved_nodes(new_nodes)

new_nodes = get_retrieved_nodes(
    "What did Gatsby want Daisy to do in front of Tom?",
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=True,
)

visualize_retrieved_nodes(new_nodes)

# ## Query Engine

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
    response_mode="tree_summarize",
)
response = query_engine.query(
    "What did the author do during his time at Y Combinator?",
)

query_engine = index.as_query_engine(
    similarity_top_k=3, response_mode="tree_summarize"
)
response = query_engine.query(
    "What did the author do during his time at Y Combinator?",
)

retrieval =

