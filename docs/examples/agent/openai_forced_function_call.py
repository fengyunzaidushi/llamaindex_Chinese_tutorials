#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/agent/openai_forced_function_call.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # OpenAI agent: specifying a forced function call

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import json
from typing import Sequence, List

from llama_index.llms import OpenAI, ChatMessage
from llama_index.tools import BaseTool, FunctionTool
from llama_index.agent import OpenAIAgent

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

def useless_tool() -> int:
    """This is a uselss tool."""
    return "This is a uselss output."

useless_tool = FunctionTool.from_defaults(fn=useless_tool)

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = OpenAIAgent.from_tools([useless_tool, add_tool], llm=llm, verbose=True)

# ### "Auto" function call

# The agent automatically selects the useful "add" tool

response = agent.chat(
    "What is 5 + 2?", tool_choice="auto"
)  # note function_call param is deprecated
# use tool_choice instead

print(response)

# ### Forced function call

# The agent is forced to call the "useless_tool" before selecting the "add" tool

response = agent.chat("What is 5 * 2?", tool_choice="useless_tool")

print(response)

# ### "None" function call

# The agent is forced to not use a tool

response = agent.chat("What is 5 * 2?", tool_choice="none")

print(response)

