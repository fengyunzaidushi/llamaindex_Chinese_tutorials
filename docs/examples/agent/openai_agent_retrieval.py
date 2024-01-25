#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/agent/openai_agent_retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Retrieval-Augmented OpenAI Agent

# to build an agent on top of OpenAI's function API and store/index an arbitrary number of tools. Our indexing/retrieval modules help to remove the complexity of having too many functions to fit in the prompt.

# #

# Let's start by importing some simple building blocks.  
# 
# The main thing we need is:
# 1. the OpenAI API
# 2. a place to keep conversation history 
# 3. a definition for tools that our agent can use.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import json
from typing import Sequence

from llama_index.tools import BaseTool, FunctionTool

# Let's define some very simple calculator tools for our agent.

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

def useless(a: int, b: int) -> int:
    """Toy useless function."""
    pass

multiply_tool = FunctionTool.from_defaults(fn=multiply, name="multiply")
useless_tools = [
    FunctionTool.from_defaults(fn=useless, name=f"useless_{str(idx)}")
    for idx in range(28)
]
add_tool = FunctionTool.from_defaults(fn=add, name="add")

all_tools = [multiply_tool] + [add_tool] + useless_tools
all_tools_map = {t.metadata.name: t for t in all_tools}

# ## Building an Object Index
# 
# We have an `ObjectIndex` construct in LlamaIndex that allows the user to use our index data structures over arbitrary objects.
# The ObjectIndex will handle serialiation to/from the object, and use an underying index (e.g. VectorStoreIndex, SummaryIndex, KeywordTableIndex) as the storage mechanism. 
# 

# 
# The index comes bundled with a retrieval mechanism, an `ObjectRetriever`. 
# 
# This can be passed in to our agent so that it can 
# perform Tool retrieval during query-time.

# define an "object" index over these tools
from llama_index import VectorStoreIndex
from llama_index.objects import ObjectIndex, SimpleToolNodeMapping

tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
obj_index = ObjectIndex.from_objects(
    all_tools,
    tool_mapping,
    VectorStoreIndex,
)

# ## Our `FnRetrieverOpenAIAgent` Implementation 

# We provide a `FnRetrieverOpenAIAgent` implementation in LlamaIndex, which can take in an `ObjectRetriever` over a set of `BaseTool` objects.
# 
# During query-time, we would first use the `ObjectRetriever` to retrieve a set of relevant Tools. These tools would then be passed into the agent; more specifically, their function signatures would be passed into the OpenAI Function calling API. 

from llama_index.agent import FnRetrieverOpenAIAgent

agent = FnRetrieverOpenAIAgent.from_retriever(
    obj_index.as_retriever(), verbose=True
)

agent.chat("What's 212 multiplied by 122? Make sure to use Tools")

agent.chat("What's 212 added to 122 ? Make sure to use Tools")

