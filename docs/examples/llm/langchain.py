#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/langchain.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## LangChain LLM

from langchain.llms import OpenAI

from llama_index.llms import LangChainLLM

llm = LangChainLLM(llm=OpenAI())

response_gen = llm.stream_complete("Hi this is")

for delta in response_gen:
    print(delta.delta, end="")

