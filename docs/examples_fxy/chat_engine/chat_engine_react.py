#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/chat_engine/chat_engine_react.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Chat Engine - ReAct Agent Mode

# ReAct is an agent based chat mode built on top of a query engine over your data.

# For each chat interaction, the agent enter a ReAct loop:
# * first decide whether to use the query engine tool and come up with appropriate input
# * (optional) use the query engine tool and observe its output
# * decide whether to repeat or give final response

# This approach is flexible, since it can flexibility choose between querying the knowledge base or not.
# However, the performance is also more dependent on the quality of the LLM. 
# You might need to do more coercing to make sure it chooses to query the knowledge base at right times, instead of hallucinating an answer.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Get started in 5 lines of code

# Load data and build index

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI, Anthropic

service_context = ServiceContext.from_defaults(llm=OpenAI())
data = SimpleDirectoryReader(input_dir="./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(data, service_context=service_context)

# Configure chat engine

chat_engine = index.as_chat_engine(chat_mode="react", verbose=True)

# Chat with your data

response = chat_engine.chat(
    "Use the tool to answer what did Paul Graham do in the summer of 1995?"
)

print(response)

# ### Customize LLM

# Use Anthropic ("claude-2")

service_context = ServiceContext.from_defaults(llm=Anthropic())

# Configure chat engine

chat_engine = index.as_chat_engine(
    service_context=service_context, chat_mode="react", verbose=True
)

response = chat_engine.chat("what did Paul Graham do in the summer of 1995?")

print(response)

response = chat_engine.chat("What did I ask you before?")

print(response)

# Reset chat engine

chat_engine.reset()

response = chat_engine.chat("What did I ask you before?")

print(response)

