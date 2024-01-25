#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/output_parsing/openai_sub_question.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # OpenAI function calling for Sub-Question Query Engine

# The sub-question query engine is designed to accept swappable question generators that implement the `BaseQuestionGenerator` interface.  
# To leverage the power of openai function calling API, we implemented a new `OpenAIQuestionGenerator` (powered by our `OpenAIPydanticProgram`)

# ## OpenAI Question Generator

# Unlike the default `LLMQuestionGenerator` that supports generic LLMs via the completion API, `OpenAIQuestionGenerator` only works with the latest OpenAI models that supports the function calling API. 
# 
# The benefit is that these models are fine-tuned to output JSON objects, so we can worry less about output parsing issues.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.question_gen.openai_generator import OpenAIQuestionGenerator

question_gen = OpenAIQuestionGenerator.from_defaults()

# Let's test it out!

from llama_index.tools import ToolMetadata
from llama_index import QueryBundle

tools = [
    ToolMetadata(
        name="march_22",
        description=(
            "Provides information about Uber quarterly financials ending March"
            " 2022"
        ),
    ),
    ToolMetadata(
        name="june_22",
        description=(
            "Provides information about Uber quarterly financials ending June"
            " 2022"
        ),
    ),
    ToolMetadata(
        name="sept_22",
        description=(
            "Provides information about Uber quarterly financials ending"
            " September 2022"
        ),
    ),
    ToolMetadata(
        name="sept_21",
        description=(
            "Provides information about Uber quarterly financials ending"
            " September 2022"
        ),
    ),
    ToolMetadata(
        name="june_21",
        description=(
            "Provides information about Uber quarterly financials ending June"
            " 2022"
        ),
    ),
    ToolMetadata(
        name="march_21",
        description=(
            "Provides information about Uber quarterly financials ending March"
            " 2022"
        ),
    ),
]

sub_questions = question_gen.generate(
    tools=tools,
    query=QueryBundle(
        "Compare the fastest growing sectors for Uber in the first two"
        " quarters of 2022"
    ),
)

sub_questions

