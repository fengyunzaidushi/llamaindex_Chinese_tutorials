#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/llm/bedrock.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Bedrock

# ## Basic Usage

# #### Call `complete` with a prompt

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.llms import Bedrock

profile_name = "Your aws profile name"
resp = Bedrock(
    model="amazon.titan-text-express-v1", profile_name=profile_name
).complete("Paul Graham is ")

print(resp)

# #### Call `chat` with a list of messages

from llama_index.llms import ChatMessage, Bedrock

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]

resp = Bedrock(
    model="amazon.titan-text-express-v1", profile_name=profile_name
).chat(messages)

print(resp)

# ## Streaming

# Using `stream_complete` endpoint 

from llama_index.llms import Bedrock

llm = Bedrock(model="amazon.titan-text-express-v1", profile_name=profile_name)
resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    print(r.delta, end="")

# Using `stream_chat` endpoint

from llama_index.llms import Bedrock

llm = Bedrock(model="amazon.titan-text-express-v1", profile_name=profile_name)
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")

# ## Configure Model

from llama_index.llms import Bedrock

llm = Bedrock(model="amazon.titan-text-express-v1", profile_name=profile_name)

resp = llm.complete("Paul Graham is ")

print(resp)

# # Connect to Bedrock with Access Keys 

from llama_index.llms import Bedrock

llm = Bedrock(
    model="amazon.titan-text-express-v1",
    aws_access_key_id="AWS Access Key ID to use",
    aws_secret_access_key="AWS Secret Access Key to use",
    aws_session_token="AWS Session Token to use",
    aws_region_name="AWS Region to use, eg. us-east-1",
)

resp = llm.complete("Paul Graham is ")

print(resp)

