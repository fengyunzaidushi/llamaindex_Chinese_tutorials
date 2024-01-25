#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/query_engine/sub_question_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Sub Question Query Engine

# It first breaks down the complex query into sub questions for each relevant data source,
# then gather all the intermediate reponses and synthesizes a final response.

# ### Preparation

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

import nest_asyncio

nest_asyncio.apply()

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index import ServiceContext

# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
service_context = ServiceContext.from_defaults(
    callback_manager=callback_manager
)

# ### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# load data
pg_essay = SimpleDirectoryReader(input_dir="./data/paul_graham/").load_data()

# build index and query engine
vector_query_engine = VectorStoreIndex.from_documents(
    pg_essay, use_async=True, service_context=service_context
).as_query_engine()

# ### Setup sub question query engine

# setup base query engine as tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="pg_essay",
            description="Paul Graham essay on What I Worked On",
        ),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    service_context=service_context,
    use_async=True,
)

# ### Run queries

response = query_engine.query(
    "How was Paul Grahams life different before, during, and after YC?"
)

print(response)

# iterate through sub_question items captured in SUB_QUESTION event
from llama_index.callbacks.schema import CBEventType, EventPayload

for i, (start_event, end_event) in enumerate(
    llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)
):
    qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
    print("Sub Question " + str(i) + ": " + qa_pair.sub_q.sub_question.strip())
    print("Answer: " + qa_pair.answer.strip())
    print("====================================")

