#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/agent/openai_agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Build your own OpenAI Agent

# With the [new OpenAI API](https://openai.com/blog/function-calling-and-other-api-updates) that supports function calling, it's never been easier to build your own agent!
# 

# #

# Let's start by importing some simple building blocks.  
# 
# The main thing we need is:
# 1. the OpenAI API (using our own `llama_index` LLM class)
# 2. a place to keep conversation history 
# 3. a definition for tools that our agent can use.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
# 

#('pip install llama-index')

import json
from typing import Sequence, List

from llama_index.llms import OpenAI, ChatMessage
from llama_index.tools import BaseTool, FunctionTool

import nest_asyncio

nest_asyncio.apply()

# Let's define some very simple calculator tools for our agent.

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

# ## Agent Definition

# Now, we define our agent that's capable of holding a conversation and calling tools in **under 50 lines of code**.
# 
# The meat of the agent logic is in the `chat` method. At a high-level, there are 3 steps:
# 1. Call OpenAI to decide which tool (if any) to call and with what arguments.
# 2. Call the tool with the arguments to obtain an output
# 3. Call OpenAI to synthesize a response from the conversation context and the tool output.
# 
# The `reset` method simply resets the conversation context, so we can start another conversation.

class YourOpenAIAgent:
    def __init__(
        self,
        tools: Sequence[BaseTool] = [],
        llm: OpenAI = OpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        chat_history: List[ChatMessage] = [],
    ) -> None:
        self._llm = llm
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history

    def reset(self) -> None:
        self._chat_history = []

    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role="user", content=message))
        tools = [
            tool.metadata.to_openai_tool() for _, tool in self._tools.items()
        ]

        ai_message = self._llm.chat(chat_history, tools=tools).message
        additional_kwargs = ai_message.additional_kwargs
        chat_history.append(ai_message)

        tool_calls = ai_message.additional_kwargs.get("tool_calls", None)
        # parallel function calling is now supported
        if tool_calls is not None:
            for tool_call in tool_calls:
                function_message = self._call_function(tool_call)
                chat_history.append(function_message)
                ai_message = self._llm.chat(chat_history).message
                chat_history.append(ai_message)

        return ai_message.content

    def _call_function(self, tool_call: dict) -> ChatMessage:
        id_ = tool_call["id"]
        function_call = tool_call["function"]
        tool = self._tools[function_call["name"]]
        output = tool(**json.loads(function_call["arguments"]))
        return ChatMessage(
            name=function_call["name"],
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": id_,
                "name": function_call["name"],
            },
        )

# ## Let's Try It Out!

agent = YourOpenAIAgent(tools=[multiply_tool, add_tool])

agent.chat("Hi")

agent.chat("What is 2123 * 215123")

# ## Our (Slightly Better) `OpenAIAgent` Implementation 

# We provide a (slightly better) `OpenAIAgent` implementation in LlamaIndex, which you can directly use as follows.  
# 

# * it implements the `BaseChatEngine` and `BaseQueryEngine` interface, so you can more seamlessly use it in the LlamaIndex framework. 
# * it supports multiple function calls per conversation turn
# * it supports streaming
# * it supports async endpoints
# * it supports callback and tracing

from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = OpenAIAgent.from_tools(
    [multiply_tool, add_tool], llm=llm, verbose=True
)

# ### Chat

response = agent.chat("What is (121 * 3) + 42?")
print(str(response))

# inspect sources
print(response.sources)

# ### Async Chat

response = await agent.achat("What is 121 * 3?")
print(str(response))

# ### Streaming Chat
# Here, every LLM response is returned as a generator. You can stream every incremental step, or only the last response.

response = agent.stream_chat(
    "What is 121 * 2? Once you have the answer, use that number to write a"
    " story about a group of mice."
)

response_gen = response.response_gen

for token in response_gen:
    print(token, end="")

# ### Async Streaming Chat

response = await agent.astream_chat(
    "What is 121 + 8? Once you have the answer, use that number to write a"
    " story about a group of mice."
)

response_gen = response.response_gen

async for token in response.async_response_gen():
    print(token, end="")

# ### Agent with Personality

# You can specify a system prompt to give the agent additional instruction or personality.

from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.prompts.system import SHAKESPEARE_WRITING_ASSISTANT

llm = OpenAI(model="gpt-3.5-turbo-0613")

agent = OpenAIAgent.from_tools(
    [multiply_tool, add_tool],
    llm=llm,
    verbose=True,
    system_prompt=SHAKESPEARE_WRITING_ASSISTANT,
)

response = agent.chat("Hi")
print(response)

response = agent.chat("Tell me a story")
print(response)

