#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/query_engine/JointQASummary.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Joint QA Summary Query Engine

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import nest_asyncio

nest_asyncio.apply()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.composability.joint_qa_summary import (
    QASummaryQueryEngineBuilder,
)
from llama_index import SimpleDirectoryReader, ServiceContext
from llama_index.response.notebook_utils import #display_response
from llama_index.llms import OpenAI

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ## Load Data

reader = SimpleDirectoryReader("./data/paul_graham/")
documents = reader.load_data()

gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4, chunk_size=1024)

chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context_chatgpt = ServiceContext.from_defaults(
    llm=chatgpt, chunk_size=1024
)

# NOTE: can also specify an existing docstore, service context, summary text, qa_text, etc.
query_engine_builder = QASummaryQueryEngineBuilder(
    service_context=service_context_gpt4
)
query_engine = query_engine_builder.build_from_documents(documents)

response = query_engine.query(
    "Can you give me a summary of the author's life?",
)

response = query_engine.query(
    "What did the author do growing up?",
)

response = query_engine.query(
    "What did the author do during his time in art school?",
)

