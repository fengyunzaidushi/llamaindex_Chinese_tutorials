#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/callbacks/LlamaDebugHandler.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Llama Debug Handler
# 
# Here we showcase the capabilities of our LlamaDebugHandler in logging events as we run queries
# within LlamaIndex.
# 
# **NOTE**: This is a beta feature. The usage within different classes and the API interface
#     for the CallbackManager and LlamaDebugHandler may change!

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
)

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

from llama_index import SimpleDirectoryReader

docs = SimpleDirectoryReader("./data/paul_graham/").load_data()

# ## Callback Manager Setup

from llama_index import ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

service_context = ServiceContext.from_defaults(
    callback_manager=callback_manager, llm=llm
)

# ## Trigger the callback with a query

from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(docs, service_context=service_context)
query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")

# ## Explore the Debug Information
# 
# The callback manager will log several start and end events for the following types:
# - CBEventType.LLM
# - CBEventType.EMBEDDING
# - CBEventType.CHUNKING
# - CBEventType.NODE_PARSING
# - CBEventType.RETRIEVE
# - CBEventType.SYNTHESIZE 
# - CBEventType.TREE
# - CBEventType.QUERY
# 
# The LlamaDebugHandler provides a few basic methods for exploring information about these events

# Print info on the LLM calls during the summary index query
print(llama_debug.get_event_time_info(CBEventType.LLM))

# Print info on llm inputs/outputs - returns start/end events for each LLM call
event_pairs = llama_debug.get_llm_inputs_outputs()
print(event_pairs[0][0])
print(event_pairs[0][1].payload.keys())
print(event_pairs[0][1].payload["response"])

# Get info on any event type
event_pairs = llama_debug.get_event_pairs(CBEventType.CHUNKING)
print(event_pairs[0][0].payload.keys())  # get first chunking start event
print(event_pairs[0][1].payload.keys())  # get first chunking end event

# Clear the currently cached events
llama_debug.flush_event_logs()

# ## See Traces & Events for Agents

# First create a tool for the agent
from llama_index.tools import QueryEngineTool

tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="PaulGrahamQuestionAnswer",
    description="Given a question about Paul Graham, will return an answer.",
)

# Now construct the agent
from llama_index.agent import OpenAIAgent

agent = OpenAIAgent.from_tools(
    tools=[tool], llm=llm, callback_manager=callback_manager
)

response = agent.chat("What did Paul do growing up?")

# works the same for async
response = await agent.achat("What did Paul do growing up?")

# Clear the currently cached events
llama_debug.flush_event_logs()

