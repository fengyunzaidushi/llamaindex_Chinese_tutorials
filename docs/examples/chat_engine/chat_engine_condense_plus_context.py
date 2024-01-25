#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/chat_engine/chat_engine_context.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 
# # Chat Engine - Condense Plus Context Mode

# This is a multi-step chat mode built on top of a retriever over your data.

# For each chat interaction:
# * First condense a conversation and latest user message to a standalone question
# * Then build a context for the standalone question from a retriever,
# * Then pass the context along with prompt and user message to LLM to generate a response.

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

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI

service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo")
)
data = SimpleDirectoryReader(input_dir="./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(data, service_context=service_context)

# Configure chat engine
# 
# Since the context retrieved can take up a large amount of the available LLM context, let's ensure we configure a smaller limit to the chat history!

from llama_index.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about an essay discussing Paul Grahams life."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=False,
)

# Chat with your data

response = chat_engine.chat("What did Paul Graham do growing up")

print(response)

# Ask a follow up question

response_2 = chat_engine.chat("Can you tell me more?")

print(response_2)

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

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about an essay discussing Paul Grahams life."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Based on the above documents, provide a detailed answer for the user question below."
    ),
)

response = chat_engine.stream_chat("What did Paul Graham do after YC?")
for token in response.response_gen:
    print(token, end="")

