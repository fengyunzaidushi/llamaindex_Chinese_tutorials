#!/usr/bin/env python
# coding: utf-8

# # BatchEvalRunner - Running Multiple Evaluations
# 
# The `BatchEvalRunner` class can be used to run a series of evaluations asynchronously. The async jobs are limited to a defined size of `num_workers`.
# 
# ## Setup

# attach to the same event-loop
import nest_asyncio

nest_asyncio.apply()

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    Response,
)
from llama_index.llms import OpenAI
from llama_index.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
)
import pandas as pd

pd.set_option("#display.max_colwidth", 0)

# Using GPT-4 here for evaluation

# gpt-4
gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

faithfulness_gpt4 = FaithfulnessEvaluator(service_context=service_context_gpt4)
relevancy_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)
correctness_gpt4 = CorrectnessEvaluator(service_context=service_context_gpt4)

documents = SimpleDirectoryReader("./test_wiki_data/").load_data()

# create vector index
llm = OpenAI(temperature=0.3, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# ## Question Generation
# 
# To run evaluations in batch, you can create the runner and then call the `.aevaluate_queries()` function on a list of queries.
# 
# First, we can generate some questions and then run evaluation on them.

#('pip install spacy datasets span-marker scikit-learn')

from llama_index.evaluation import DatasetGenerator

dataset_generator = DatasetGenerator.from_documents(
    documents, service_context=service_context
)

qas = dataset_generator.generate_dataset_from_nodes(num=3)

# ## Running Batch Evaluation
# 
# Now, we can run our batch evaluation!

from llama_index.evaluation import BatchEvalRunner

runner = BatchEvalRunner(
    {"faithfulness": faithfulness_gpt4, "relevancy": relevancy_gpt4},
    workers=8,
)

eval_results = await runner.aevaluate_queries(
    vector_index.as_query_engine(), queries=qas.questions
)

# If we had ground-truth answers, we could also include the correctness evaluator like below.
# The correctness evaluator depends on additional kwargs, which are passed in as a dictionary.
# Each question is mapped to a set of kwargs
#

# runner = BatchEvalRunner(
#     {"correctness": correctness_gpt4},
#     workers=8,
# )

# eval_results = await runner.aevaluate_queries(
#     vector_index.as_query_engine(),
#     queries=qas.queries,
#     reference=[qr[1] for qr in qas.qr_pairs],
# )

print(len([qr for qr in qas.qr_pairs]))

# #

print(eval_results.keys())

print(eval_results["correctness"][0].dict().keys())

print(eval_results["correctness"][0].passing)
print(eval_results["correctness"][0].response)
print(eval_results["correctness"][0].contexts)

# ## Reporting Total Scores

def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score

score = get_eval_results("correctness", eval_results)

score = get_eval_results("relevancy", eval_results)

