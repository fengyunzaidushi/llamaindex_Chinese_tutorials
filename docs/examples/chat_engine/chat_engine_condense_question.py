#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/chat_engine/chat_engine_condense_question.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 
# # Chat Engine - Condense Question Mode

# Condense question is a simple chat mode built on top of a query engine over your data.

# For each chat interaction:
# * first generate a standalone question from conversation context and last message, then 
# * query the query engine with the condensed question for a response.

# This approach is simple, and works for questions directly related to the knowledge base. 
# Since it *always* queries the knowledge base, it can have difficulty answering meta questions like "what did I ask you before?"

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ## Get started in 5 lines of code

# Load data and build index

from llama_index import VectorStoreIndex, SimpleDirectoryReader

data = SimpleDirectoryReader(input_dir="./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(data)

# Configure chat engine

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Chat with your data

response = chat_engine.chat("What did Paul Graham do after YC?")

print(response)

# Ask a follow up question

response = chat_engine.chat("What about after that?")

print(response)

response = chat_engine.chat("Can you tell me more?")

print(response)

# Reset conversation state

chat_engine.reset()

response = chat_engine.chat("What about after that?")

print(response)

# ## Streaming Support

from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI

service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0)
)

data = SimpleDirectoryReader(input_dir="../data/paul_graham/").load_data()

index = VectorStoreIndex.from_documents(data, service_context=service_context)

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

response = chat_engine.stream_chat("What did Paul Graham do after YC?")
for token in response.response_gen:
    print(token, end="")

