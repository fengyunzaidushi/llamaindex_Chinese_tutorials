#!/usr/bin/env python
# coding: utf-8

# # Single-Turn Multi-Function Calling OpenAI Agents
# 
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/agent/openai_agent_parallel_function_calling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# With the latest OpenAI API (v. 1.1.0+), users can now execute multiple function calls within a single turn of `User` and `Agent` dialogue. We've updated our library to enable this new feature as well, and in this notebook we'll show you how it all works!
# 
# NOTE: OpenAI refers to this as "Parallel" function calling, but the current implementation doesn't invoke parallel computations of the multiple function calls. So, it's "parallelizable" function calling in terms of our current implementation.

from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.tools import BaseTool, FunctionTool

# ### Setup
# 
# If you've seen any of our previous notebooks on OpenAI Agents, then you're already familiar with the cookbook recipe that we have to follow here. But if not, or if you fancy a refresher then all we need to do (at a high level) are the following steps:
# 
# 1. Define a set of tools (we'll use `FunctionTool`) since Agents work with tools
# 2. Define the `LLM` for the Agent
# 3. Define a `OpenAIAgent`

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

llm = OpenAI(model="gpt-3.5-turbo-1106")
agent = OpenAIAgent.from_tools(
    [multiply_tool, add_tool], llm=llm, verbose=True
)

# ### Sync mode

response = agent.chat("What is (121 * 3) + 42?")
print(str(response))

response = agent.stream_chat("What is (121 * 3) + 42?")

# ### Async mode

import nest_asyncio

nest_asyncio.apply()

response = await agent.achat("What is (121 * 3) + 42?")
print(str(response))

response = await agent.astream_chat("What is (121 * 3) + 42?")

response_gen = response.response_gen

async for token in response.async_response_gen():
    print(token, end="")

# ### Example from OpenAI docs
# 
# Here's an example straight from the OpenAI [docs](https://platform.openai.com/docs/guides/function-calling/parallel-function-calling) on Parallel function calling. (Their example gets this done in 76 lines of code, whereas with the `llama_index` library you can get that down to about 18 lines.)

import json

# Example dummy function hard coded to return the same weather

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps(
            {"location": location, "temperature": "10", "unit": "celsius"}
        )
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": location, "temperature": "72", "unit": "fahrenheit"}
        )
    else:
        return json.dumps(
            {"location": location, "temperature": "22", "unit": "celsius"}
        )

weather_tool = FunctionTool.from_defaults(fn=get_current_weather)

llm = OpenAI(model="gpt-3.5-turbo-1106")
agent = OpenAIAgent.from_tools([weather_tool], llm=llm, verbose=True)
response = agent.chat(
    "What's the weather like in San Francisco, Tokyo, and Paris?"
)

# All of the above function calls that the Agent has done above were in a single turn of dialogue between the `Assistant` and the `User`. What's interesting is that an older version of GPT-3.5 is not quite advanced enough compared to is successor — it will do the above task in 3 separate turns. For the sake of demonstration, here it is below.

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = OpenAIAgent.from_tools([weather_tool], llm=llm, verbose=True)
response = agent.chat(
    "What's the weather like in San Francisco, Tokyo, and Paris?"
)

# ## Conclusion
# 
# And so, as you can see the `llama_index` library can handle multiple function calls (as well as a single function call) within a single turn of dialogue between the user and the OpenAI agent!
