#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/evaluation/guideline_eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Guideline Evaluator

# This notebook shows how to use `GuidelineEvaluator` to evaluate a question answer system given user specified guidelines.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.evaluation import GuidelineEvaluator
from llama_index import ServiceContext
from llama_index.llms import OpenAI

# Needed for running async functions in Jupyter Notebook
import nest_asyncio

nest_asyncio.apply()

GUIDELINES = [
    "The response should fully answer the query.",
    "The response should avoid being vague or ambiguous.",
    (
        "The response should be specific and use statistics or numbers when"
        " possible."
    ),
]

service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4"))

evaluators = [
    GuidelineEvaluator(service_context=service_context, guidelines=guideline)
    for guideline in GUIDELINES
]

sample_data = {
    "query": "Tell me about global warming.",
    "contexts": [
        (
            "Global warming refers to the long-term increase in Earth's"
            " average surface temperature due to human activities such as the"
            " burning of fossil fuels and deforestation."
        ),
        (
            "It is a major environmental issue with consequences such as"
            " rising sea levels, extreme weather events, and disruptions to"
            " ecosystems."
        ),
        (
            "Efforts to combat global warming include reducing carbon"
            " emissions, transitioning to renewable energy sources, and"
            " promoting sustainable practices."
        ),
    ],
    "response": (
        "Global warming is a critical environmental issue caused by human"
        " activities that lead to a rise in Earth's temperature. It has"
        " various adverse effects on the planet."
    ),
}

for guideline, evaluator in zip(GUIDELINES, evaluators):
    eval_result = evaluator.evaluate(
        query=sample_data["query"],
        contexts=sample_data["contexts"],
        response=sample_data["response"],
    )
    print("=====")
    print(f"Guideline: {guideline}")
    print(f"Pass: {eval_result.passing}")
    print(f"Feedback: {eval_result.feedback}")

