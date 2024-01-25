#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/chat_engine/chat_engine_context.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 
# # Chat Engine - Context Mode

# ContextChatEngine is a simple chat mode built on top of a retriever over your data.

# For each chat interaction:
# * first retrieve text from the index using the user message
# * set the retrieved text as context in the system prompt
# * return an answer to the user message

# This approach is simple, and works for questions directly related to the knowledge base and general interactions.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ## Get started in 5 lines of code

# Load data and build index

import openai
import os

os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index import VectorStoreIndex, SimpleDirectoryReader

data = SimpleDirectoryReader(input_dir="./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(data)

# Configure chat engine
# 
# Since the context retrieved can take up a large amount of the available LLM context, let's ensure we configure a smaller limit to the chat history!

from llama_index.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about an essay discussing Paul Grahams life."
    ),
)

# Chat with your data

response = chat_engine.chat("Hello!")

print(response)

# Ask a follow up question

response = chat_engine.chat("What did Paul Graham do growing up?")

print(response)

response = chat_engine.chat("Can you tell me more?")

print(response)

# Reset conversation state

chat_engine.reset()

response = chat_engine.chat("Hello! What do you know?")

print(response)

# ## Streaming Support

from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    set_global_service_context,
)
from llama_index.llms import OpenAI

service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0)
)
set_global_service_context(service_context)

data = SimpleDirectoryReader(input_dir="./data/paul_graham/").load_data()

index = VectorStoreIndex.from_documents(data)

chat_engine = index.as_chat_engine(chat_mode="context")

response = chat_engine.stream_chat("What did Paul Graham do after YC?")
for token in response.response_gen:
    print(token, end="")

