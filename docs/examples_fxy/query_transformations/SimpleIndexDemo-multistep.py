#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/query_transformations/SimpleIndexDemo-multistep.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Multi-Step Query Engine
# 
# We have a multi-step query engine that's able to decompose a complex query into sequential subquestions. This
# guide walks you through how to set it up!

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Load documents, build the VectorStoreIndex

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.llms import OpenAI
from IPython.#display import Markdown, #display

# LLM (gpt-3)
gpt3 = OpenAI(temperature=0, model="text-davinci-003")
service_context_gpt3 = ServiceContext.from_defaults(llm=gpt3)

# LLM (gpt-4)
gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

index = VectorStoreIndex.from_documents(documents)

# #### Query Index

from llama_index.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)

# gpt-4
step_decompose_transform = StepDecomposeQueryTransform(llm=gpt4, verbose=True)

# gpt-3
step_decompose_transform_gpt3 = StepDecomposeQueryTransform(
    llm=gpt3, verbose=True
)

index_summary = "Used to answer questions about the author"

# set Logging to DEBUG for more detailed outputs
from llama_index.query_engine.multistep_query_engine import (
    MultiStepQueryEngine,
)

query_engine = index.as_query_engine(service_context=service_context_gpt4)
query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_decompose_transform,
    index_summary=index_summary,
)
response_gpt4 = query_engine.query(
    "Who was in the first batch of the accelerator program the author"
    " started?",
)

#display(Markdown(f"<b>{response_gpt4}</b>"))

sub_qa = response_gpt4.metadata["sub_qa"]
tuples = [(t[0], t[1].response) for t in sub_qa]
print(tuples)

response_gpt4 = query_engine.query(
    "In which city did the author found his first company, Viaweb?",
)

print(response_gpt4)

query_engine = index.as_query_engine(service_context=service_context_gpt3)
query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_decompose_transform_gpt3,
    index_summary=index_summary,
)

response_gpt3 = query_engine.query(
    "In which city did the author found his first company, Viaweb?",
)

print(response_gpt3)

