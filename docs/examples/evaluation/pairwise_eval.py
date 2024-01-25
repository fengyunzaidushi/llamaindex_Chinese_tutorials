#!/usr/bin/env python
# coding: utf-8

# # Pairwise Evaluator
# 
# This notebook uses the `PairwiseEvaluator` module to see if an evaluation LLM would prefer one query engine over another.  

# attach to the same event-loop
import nest_asyncio

nest_asyncio.apply()

# configuring logger to INFO level
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    Response,
)
from llama_index.llms import OpenAI
from llama_index.evaluation import PairwiseComparisonEvaluator
import pandas as pd

pd.set_option("#display.max_colwidth", 0)

# Using GPT-4 here for evaluation

# gpt-4
gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

evaluator_gpt4 = PairwiseComparisonEvaluator(
    service_context=service_context_gpt4
)

documents = SimpleDirectoryReader("./test_wiki_data/").load_data()

# create vector index
service_context1 = ServiceContext.from_defaults(chunk_size=512)
vector_index1 = VectorStoreIndex.from_documents(
    documents, service_context=service_context1
)

service_context2 = ServiceContext.from_defaults(chunk_size=128)
vector_index2 = VectorStoreIndex.from_documents(
    documents, service_context=service_context2
)

query_engine1 = vector_index1.as_query_engine(similarity_top_k=2)
query_engine2 = vector_index2.as_query_engine(similarity_top_k=8)

# define jupyter #display function
def #display_eval_df(query, response1, response2, eval_result) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Reference Response (Answer 1)": response2,
            "Current Response (Answer 2)": response1,
            "Score": eval_result.score,
            "Reason": eval_result.feedback,
        },
        index=[0],
    )
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "300px",
            "overflow-wrap": "break-word",
        },
        subset=["Current Response (Answer 2)", "Reference Response (Answer 1)"]
    )
    #display(eval_df)

# To run evaluations you can call the `.evaluate_response()` function on the `Response` object return from the query to run the evaluations. Lets evaluate the outputs of the vector_index.

# query_str = "How did New York City get its name?"
query_str = "What was the role of NYC during the American Revolution?"
# query_str = "Tell me about the arts and culture of NYC"
response1 = str(query_engine1.query(query_str))
response2 = str(query_engine2.query(query_str))

# By default, we enforce "consistency" in the pairwise comparison.
# 
# We try feeding in the candidate, reference pair, and then swap the order of the two, and make sure that the results are still consistent (or return a TIE if not).

eval_result = await evaluator_gpt4.aevaluate(
    query_str, response=response1, reference=response2
)

#display_eval_df(query_str, response1, response2, eval_result)

# **NOTE**: By default, we enforce consensus by flipping the order of response/reference and making sure that the answers are opposites.
# 
# We can disable this - which can lead to more inconsistencies!

evaluator_gpt4_nc = PairwiseComparisonEvaluator(
    service_context=service_context_gpt4, enforce_consensus=False
)

eval_result = await evaluator_gpt4_nc.aevaluate(
    query_str, response=response1, reference=response2
)

#display_eval_df(query_str, response1, response2, eval_result)

eval_result = await evaluator_gpt4_nc.aevaluate(
    query_str, response=response2, reference=response1
)

#display_eval_df(query_str, response2, response1, eval_result)

# ## Running on some more Queries

query_str = "Tell me about the arts and culture of NYC"
response1 = str(query_engine1.query(query_str))
response2 = str(query_engine2.query(query_str))

eval_result = await evaluator_gpt4.aevaluate(
    query_str, response=response1, reference=response2
)

#display_eval_df(query_str, response1, response2, eval_result)

