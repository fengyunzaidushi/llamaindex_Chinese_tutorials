#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/customization/streaming/chat_engine_condense_question_stream_response.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Streaming for Chat Engine - Condense Question Mode

# Load documents, build the VectorStoreIndex

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

index = VectorStoreIndex.from_documents(documents)

# Chat with your data

chat_engine = index.as_chat_engine(
    chat_mode="condense_question", streaming=True
)
response_stream = chat_engine.chat("What did Paul Graham do after YC?")

response_stream.print_response_stream()

# Ask a follow up question

response_stream = chat_engine.chat("What about after that?")

response_stream.print_response_stream()

response_stream = chat_engine.chat("Can you tell me more?")

response_stream.print_response_stream()

# Reset conversation state

chat_engine.reset()

response_stream = chat_engine.chat("What about after that?")

response_stream.print_response_stream()

