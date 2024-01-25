#!/usr/bin/env python
# coding: utf-8

# # WatsonX

# ## Basic Usage

# #### Call `complete` with a prompt

from llama_index.llms import WatsonX

credentials = {
    "url": "https://enter.your-ibm.url",
    "apikey": "insert_your_api_key",
}

project_id = "insert_your_project_id"

resp = WatsonX(credentials=credentials, project_id=project_id).complete(
    "Paul Graham is"
)

print(resp)

# #### Call `chat` with a list of messages

from llama_index.llms import ChatMessage, WatsonX

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]

resp = WatsonX(
    model_id="meta-llama/llama-2-70b-chat",
    credentials=credentials,
    project_id=project_id,
).chat(messages)

print(resp)

# ## Streaming

#  Using `stream_complete` endpoint

from llama_index.llms import WatsonX

llm = WatsonX(credentials=credentials, project_id=project_id)

resp = llm.stream_complete("Paul Graham is")

for r in resp:
    print(r.delta, end="")

# Using `stream_chat` endpoint

from llama_index.llms import WatsonX

llm = WatsonX(
    model_id="meta-llama/llama-2-70b-chat",
    credentials=credentials,
    project_id=project_id,
)
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

from llama_index.llms import WatsonX

llm = WatsonX(
    model_id="meta-llama/llama-2-70b-chat",
    credentials=credentials,
    project_id=project_id,
    temperature=0,
    max_new_tokens=100,
)

resp = llm.complete("Paul Graham is")

print(resp)

