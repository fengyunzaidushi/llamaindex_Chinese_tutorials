#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/evaluation/QuestionGeneration.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # QuestionGeneration
# 
# This notebook walks through the process of generating a list of questions that could be asked about your data. This is useful for setting up an evaluation pipeline using the `FaithfulnessEvaluator` and `RelevancyEvaluator` evaluation tools.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    Response,
)
from llama_index.llms import OpenAI

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# Load Data

reader = SimpleDirectoryReader("./data/paul_graham/")
documents = reader.load_data()

data_generator = DatasetGenerator.from_documents(documents)

eval_questions = data_generator.generate_questions_from_nodes()

eval_questions

# gpt-4
gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

evaluator_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)

# create vector index
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context_gpt4
)

# define jupyter #display function
def #display_eval_df(query: str, response: Response, eval_result: str) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Response": str(response),
            "Source": (
                response.source_nodes[0].node.get_content()[:1000] + "..."
            ),
            "Evaluation Result": eval_result,
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

query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(eval_questions[1])
eval_result = evaluator_gpt4.evaluate_response(
    query=eval_questions[1], response=response_vector
)

#display_eval_df(eval_questions[1], response_vector, eval_result)

