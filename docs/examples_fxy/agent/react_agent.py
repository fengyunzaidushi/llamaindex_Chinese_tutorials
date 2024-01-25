#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/agent/react_agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # ReAct Agent

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI, ChatMessage
from llama_index.tools import BaseTool, FunctionTool

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

# ## gpt-3.5-turbo-0613

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

response = agent.chat("What is 20+2*4? Calculate step by step ")

response_gen = agent.stream_chat("What is 20+2*4? Calculate step by step")
response_gen.print_response_stream()

# ## gpt-4

llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

response = agent.chat("What is 2+2*4")
print(response)

# ## text-davinci-003

llm = OpenAI(model="text-davinci-003")
agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

response = agent.chat("What is 2+2*4")
print(response)

