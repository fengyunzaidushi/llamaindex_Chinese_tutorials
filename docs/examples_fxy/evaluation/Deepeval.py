#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/evaluation/Deepeval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # LlamaIndex + DeepEval Integration
# 
# This code tutorial shows how you can easily integrate LlamaIndex with DeepEval. DeepEval makes it easy to unit-test your LLMs.
# 
# You can read more about the DeepEval framework here: https://docs.confident-ai.com/docs/framework
# 
# Feel free to check out our repository here: https://github.com/confident-ai/deepeval
# 
# ![Framework](https://docs.confident-ai.com/assets/images/llm-evaluation-framework-example-b02144720026b6d49b1e04d8a99d3d33.png)

# ### Set-up and Installation
# 
# We recommend setting up and installing via pip!

#('pip install -q -q llama-index')
#('pip install -U -q deepeval')

# This step is optional and only if you want a server-hosted dashboard! (Psst I think you should!)

#('deepeval login')

# ### Testing for factual consistency

from llama_index.response.schema import Response
from typing import List
from llama_index.schema import Document
from deepeval.metrics.factual_consistency import FactualConsistencyMetric

# ## Setting Up The Evaluator
# 
# Setting up the evaluator.

from llama_index import (
    TreeIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    Response,
)
from llama_index.llms import OpenAI
from llama_index.evaluation import FaithfulnessEvaluator

import os
import openai

api_key = "sk-XXX"
openai.api_key = api_key

gpt4 = OpenAI(temperature=0, model="gpt-4", api_key=api_key)
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)
evaluator_gpt4 = FaithfulnessEvaluator(service_context=service_context_gpt4)

# ### Getting a LlamaHub Loader

from llama_index import download_loader

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()
documents = loader.load_data(pages=["Tokyo"])

tree_index = TreeIndex.from_documents(documents=documents)
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context_gpt4
)

# We then build an evaluator based on the `BaseEvaluator` class that requires an `evaluate` method.
# 

from typing import Any, Optional, Sequence
from llama_index.evaluation.base import BaseEvaluator, EvaluationResult

class FactualConsistencyEvaluator(BaseEvaluator):
    def evaluate(
        self,
        query: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        response: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate factual consistency metrics"""
        if response is None or contexts is None:
            raise ValueError('Please provide "response" and "contexts".')
        metric = FactualConsistencyMetric()
        context = " ".join([d for d in contexts])
        score = metric.measure(output=response, context=context)
        return EvaluationResult(
            response=response,
            contexts=contexts,
            passing=metric.is_successful(),
            score=score,
        )

evaluator = FactualConsistencyEvaluator()

query_engine = tree_index.as_query_engine()
response = query_engine.query("How did Tokyo get its name?")
eval_result = evaluator.evaluate_response(response=response)

# ## Other Metrics
# 
# We recommend using other metrics to help give more confidence to various prompt iterations, LLM outputs etc. We think ML-assisted approaches are required to give performance for these models.
# 
# - Overall Score: https://docs.confident-ai.com/docs/measuring_llm_performance/overall_score
# - Answer Relevancy: https://docs.confident-ai.com/docs/measuring_llm_performance/answer_relevancy
# - Bias: https://docs.confident-ai.com/docs/measuring_llm_performance/debias
