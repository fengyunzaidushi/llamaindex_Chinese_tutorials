#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/llm_predictor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # LLM Predictor

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## LangChain LLM

from langchain.chat_models import ChatAnyscale, ChatOpenAI
from llama_index.llms import LangChainLLM
from llama_index.prompts import PromptTemplate

llm = LangChainLLM(ChatOpenAI())

stream = await llm.astream(PromptTemplate("Hi, write a short story"))

async for token in stream:
    print(token, end="")

## Test with ChatAnyscale
llm = LangChainLLM(ChatAnyscale())

stream = llm.stream(
    PromptTemplate("Hi, Which NFL team have most Super Bowl wins")
)
for token in stream:
    print(token, end="")

# ## OpenAI LLM

from llama_index.llms import OpenAI

llm = OpenAI()

stream = await llm.astream("Hi, write a short story")

for token in stream:
    print(token, end="")

