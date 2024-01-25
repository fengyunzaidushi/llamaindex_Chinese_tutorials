#!/usr/bin/env python
# coding: utf-8

# # Relevancy Evaluator
# 
# This notebook uses the `RelevancyEvaluator` to measure if the response + source nodes match the query.  
# This is useful for measuring if the query was actually answered by the response.

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    TreeIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    Response,
)
from llama_index.llms import OpenAI
from llama_index.evaluation import RelevancyEvaluator
import pandas as pd

pd.set_option("#display.max_colwidth", 0)

# gpt-3 (davinci)
gpt3 = OpenAI(temperature=0, model="text-davinci-003")
service_context_gpt3 = ServiceContext.from_defaults(llm=gpt3)

# gpt-4
gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

evaluator = RelevancyEvaluator(service_context=service_context_gpt3)
evaluator_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)

documents = SimpleDirectoryReader("./test_wiki_data").load_data()

# create vector index
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=ServiceContext.from_defaults(chunk_size=512)
)

# define jupyter #display function
def #display_eval_df(query: str, response: Response, eval_result: str) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Response": str(response),
            "Source": response.source_nodes[0].node.text[:1000] + "...",
            "Evaluation Result": "Pass" if eval_result.passing else "Fail",
        },
        index=[0],
    )
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Response", "Source"]
    )
    #display(eval_df)

# ### Evaluate Response
# 
# Evaluate response relative to source nodes as well as query.

query_str = (
    "What battles took place in New York City in the American Revolution?"
)
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(query_str)
eval_result = evaluator_gpt4.evaluate_response(
    query=query_str, response=response_vector
)

#display_eval_df(query_str, response_vector, eval_result)

query_str = "What are the airports in New York City?"
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(query_str)
eval_result = evaluator_gpt4.evaluate_response(
    query=query_str, response=response_vector
)

#display_eval_df(query_str, response_vector, eval_result)

query_str = "Who is the mayor of New York City?"
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(query_str)
eval_result = evaluator_gpt4.evaluate_response(
    query=query_str, response=response_vector
)

#display_eval_df(query_str, response_vector, eval_result)

# ### Evaluate Source Nodes
# 
# Evaluate the set of returned sources, and determine which sources actually contain the answer to a given query.

from typing import List

# define jupyter #display function
def #display_eval_sources(
    query: str, response: Response, eval_result: List[str]
) -> None:
    sources = [s.node.get_text() for s in response.source_nodes]
    eval_df = pd.DataFrame(
        {
            "Source": sources,
            "Eval Result": eval_result,
        },
    )
    eval_df.style.set_caption(query)
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Source"]
    )

    #display(eval_df)

# NOTE: you can set response_mode="no_text" to get just the sources
query_str = "What are the airports in New York City?"
query_engine = vector_index.as_query_engine(
    similarity_top_k=3, response_mode="no_text"
)
response_vector = query_engine.query(query_str)
eval_source_result_full = [
    evaluator_gpt4.evaluate(
        query=query_str,
        response=response_vector.response,
        contexts=[source_node.get_content()],
    )
    for source_node in response_vector.source_nodes
]
eval_source_result = [
    "Pass" if result.passing else "Fail" for result in eval_source_result_full
]

#display_eval_sources(query_str, response_vector, eval_source_result)

# NOTE: you can set response_mode="no_text" to get just the sources
query_str = "Who is the mayor of New York City?"
query_engine = vector_index.as_query_engine(
    similarity_top_k=3, response_mode="no_text"
)
eval_source_result_full = [
    evaluator_gpt4.evaluate(
        query=query_str,
        response=response_vector.response,
        contexts=[source_node.get_content()],
    )
    for source_node in response_vector.source_nodes
]
eval_source_result = [
    "Pass" if result.passing else "Fail" for result in eval_source_result_full
]

#display_eval_sources(query_str, response_vector, eval_source_result)

