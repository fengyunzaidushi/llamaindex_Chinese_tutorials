#!/usr/bin/env python
# coding: utf-8

# # Faithfulness Evaluator
# 
# This notebook uses the `FaithfulnessEvaluator` module to measure if the response from a query engine matches any source nodes.  
# This is useful for measuring if the response was hallucinated.  
# The data is extracted from the [New York City](https://en.wikipedia.org/wiki/New_York_City) wikipedia page.

# attach to the same event-loop
import nest_asyncio

nest_asyncio.apply()

# configuring logger to INFO level
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
from llama_index.evaluation import FaithfulnessEvaluator
import pandas as pd

pd.set_option("#display.max_colwidth", 0)

# Using GPT-4 here for evaluation

# gpt-4
gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

evaluator_gpt4 = FaithfulnessEvaluator(service_context=service_context_gpt4)

documents = SimpleDirectoryReader("./test_wiki_data/").load_data()

# create vector index
service_context = ServiceContext.from_defaults(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# define jupyter #display function
def #display_eval_df(response: Response, eval_result: str) -> None:
    if response.source_nodes == []:
        print("no response!")
        return
    eval_df = pd.DataFrame(
        {
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

# To run evaluations you can call the `.evaluate_response()` function on the `Response` object return from the query to run the evaluations. Lets evaluate the outputs of the vector_index.

query_engine = vector_index.as_query_engine()
response_vector = query_engine.query("How did New York City get its name?")
eval_result = evaluator_gpt4.evaluate_response(response=response_vector)

#display_eval_df(response_vector, eval_result)

# ## Benchmark on Generated Question
# 
# Now lets generate a few more questions so that we have more to evaluate with and run a small benchmark.

from llama_index.evaluation import DatasetGenerator

question_generator = DatasetGenerator.from_documents(documents)
eval_questions = question_generator.generate_questions_from_nodes(5)

eval_questions

import asyncio

def evaluate_query_engine(query_engine, questions):
    c = [query_engine.aquery(q) for q in questions]
    results = asyncio.run(asyncio.gather(*c))
    print("finished query")

    total_correct = 0
    for r in results:
        # evaluate with gpt 4
        eval_result = (
            1 if evaluator_gpt4.evaluate_response(response=r).passing else 0
        )
        total_correct += eval_result

    return total_correct, len(results)

vector_query_engine = vector_index.as_query_engine()
correct, total = evaluate_query_engine(vector_query_engine, eval_questions[:5])

print(f"score: {correct}/{total}")

