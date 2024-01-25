#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/evaluation/RetryQuery.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Self Correcting Query Engines - Evaluation & Retry

# They leverage the latest LLM's ability to evaluate its own output, and then self-correct to give better responses. 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# Uncomment to add your OpenAI API key
# import os
# os.environ['OPENAI_API_KEY'] = "INSERT OPENAI KEY"

# Uncomment for debug level logging
# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# ## Setup

# First we ingest the document.

from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.readers.file.base import SimpleDirectoryReader

# Needed for running async functions in Jupyter Notebook
import nest_asyncio

nest_asyncio.apply()

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# Load Data

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents)
query = "What did the author do growing up?"

# Let's what the response from the default query engine looks like

base_query_engine = index.as_query_engine()
response = base_query_engine.query(query)
print(response)

# ## Retry Query Engine

# The retry query engine uses an evaluator to improve the response from a base query engine.  
# 
# It does the following:
# 1. first queries the base query engine, then
# 2. use the evaluator to decided if the response passes.
# 3. If the response passes, then return response,
# 4. Otherwise, transform the original query with the evaluation result (query, response, and feedback) into a new query, 
# 5. Repeat up to max_retries

from llama_index.query_engine import RetryQueryEngine
from llama_index.evaluation import RelevancyEvaluator

query_response_evaluator = RelevancyEvaluator()
retry_query_engine = RetryQueryEngine(
    base_query_engine, query_response_evaluator
)
retry_response = retry_query_engine.query(query)
print(retry_response)

# ## Retry Source Query Engine

# The Source Retry modifies the query source nodes by filtering the existing source nodes for the query based on llm node evaluation.

from llama_index.query_engine import RetrySourceQueryEngine

retry_source_query_engine = RetrySourceQueryEngine(
    base_query_engine, query_response_evaluator
)
retry_source_response = retry_source_query_engine.query(query)
print(retry_source_response)

# ## Retry Guideline Query Engine

# This module tries to use guidelines to direct the evaluator's behavior. You can customize your own guidelines.

from llama_index.evaluation.guideline import (
    GuidelineEvaluator,
    DEFAULT_GUIDELINES,
)
from llama_index.response.schema import Response
from llama_index.indices.query.query_transform.feedback_transform import (
    FeedbackQueryTransformation,
)
from llama_index.query_engine.retry_query_engine import (
    RetryGuidelineQueryEngine,
)

# Guideline eval
guideline_eval = GuidelineEvaluator(
    guidelines=DEFAULT_GUIDELINES
    + "\nThe response should not be overly long.\n"
    "The response should try to summarize where possible.\n"
)  # just for example

# Let's look like what happens under the hood.

typed_response = (
    response if isinstance(response, Response) else response.get_response()
)
eval = guideline_eval.evaluate_response(query, typed_response)
print(f"Guideline eval evaluation result: {eval.feedback}")

feedback_query_transform = FeedbackQueryTransformation(resynthesize_query=True)
transformed_query = feedback_query_transform.run(query, {"evaluation": eval})
print(f"Transformed query: {transformed_query.query_str}")

# Now let's run the full query engine

retry_guideline_query_engine = RetryGuidelineQueryEngine(
    base_query_engine, guideline_eval, resynthesize_query=True
)
retry_guideline_response = retry_guideline_query_engine.query(query)
print(retry_guideline_response)

