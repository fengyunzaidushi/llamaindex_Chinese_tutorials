#!/usr/bin/env python
# coding: utf-8

# # OpenAI Assistant Agent
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/agent/openai_assistant_agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# This shows you how to use our agent abstractions built on top of the [OpenAI Assistant API](https://platform.openai.com/docs/assistants/overview).
# 

#('pip install llama-index')

# ## Simple Agent (no external tools)
# 
# Here we show a simple example with the built-in code interpreter.

# Let's start by importing some simple building blocks.  

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
# 

from llama_index.agent import OpenAIAssistantAgent

agent = OpenAIAssistantAgent.from_new(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    openai_tools=[{"type": "code_interpreter"}],
    instructions_prefix="Please address the user as Jane Doe. The user has a premium account.",
)

agent.thread_id

response = agent.chat(
    "I need to solve the equation `3x + 11 = 14`. Can you help me?"
)

print(str(response))

# ## Assistant with Built-In Retrieval
# 
# Let's test the assistant by having it use the built-in OpenAI Retrieval tool over a user-uploaded file.
# 
# Here, we upload and pass in the file during assistant-creation time. 
# 
# The other option is you can upload/pass the file-id in for a message in a given thread with `upload_files` and `add_message`.

from llama_index.agent import OpenAIAssistantAgent

agent = OpenAIAssistantAgent.from_new(
    name="SEC Analyst",
    instructions="You are a QA assistant designed to analyze sec filings.",
    openai_tools=[{"type": "retrieval"}],
    instructions_prefix="Please address the user as Jerry.",
    files=["data/10k/lyft_2021.pdf"],
    verbose=True,
)

response = agent.chat("What was Lyft's revenue growth in 2021?")

print(str(response))

# ## Assistant with Query Engine Tools
# 
# Here we showcase the function calling capabilities of the OpenAIAssistantAgent by integrating it with our query engine tools over different documents.

# ### 1. Setup: Load Data

from llama_index.agent import OpenAIAssistantAgent
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.tools import QueryEngineTool, ToolMetadata

try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/lyft"
    )
    lyft_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/uber"
    )
    uber_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False

#("mkdir -p 'data/10k/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'")

if not index_loaded:
    # load data
    lyft_docs = SimpleDirectoryReader(
        input_files=["./data/10k/lyft_2021.pdf"]
    ).load_data()
    uber_docs = SimpleDirectoryReader(
        input_files=["./data/10k/uber_2021.pdf"]
    ).load_data()

    # build index
    lyft_index = VectorStoreIndex.from_documents(lyft_docs)
    uber_index = VectorStoreIndex.from_documents(uber_docs)

    # persist index
    lyft_index.storage_context.persist(persist_dir="./storage/lyft")
    uber_index.storage_context.persist(persist_dir="./storage/uber")

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

# ### 2. Let's Try it Out

agent = OpenAIAssistantAgent.from_new(
    name="SEC Analyst",
    instructions="You are a QA assistant designed to analyze sec filings.",
    tools=query_engine_tools,
    instructions_prefix="Please address the user as Jerry.",
    verbose=True,
    run_retrieve_sleep_time=1.0,
)

response = agent.chat("What was Lyft's revenue growth in 2021?")

# ## Assistant Agent with your own Vector Store / Retrieval API
# 
# LlamaIndex has 35+ vector database integrations. Instead of using the in-house Retrieval API, you can use our assistant agent over any vector store.
# 
# Here is our full [list of vector store integrations](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html). We picked one vector store (Supabase) using a random number generator.

from llama_index.agent import OpenAIAssistantAgent
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores import SupabaseVectorStore

from llama_index.tools import QueryEngineTool, ToolMetadata

#("mkdir -p 'data/10k/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'")

# load data
reader = SimpleDirectoryReader(input_files=["./data/10k/lyft_2021.pdf"])
docs = reader.load_data()
for doc in docs:
    doc.id_ = "lyft_docs"

vector_store = SupabaseVectorStore(
    postgres_connection_string=(
        "postgresql://<user>:<password>@<host>:<port>/<db_name>"
    ),
    collection_name="base_demo",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

# sanity check that the docs are in the vector store
num_docs = vector_store.get_by_id("lyft_docs", limit=1000)
print(len(num_docs))

lyft_tool = QueryEngineTool(
    query_engine=index.as_query_engine(similarity_top_k=3),
    metadata=ToolMetadata(
        name="lyft_10k",
        description=(
            "Provides information about Lyft financials for year 2021. "
            "Use a detailed plain text question as input to the tool."
        ),
    ),
)

agent = OpenAIAssistantAgent.from_new(
    name="SEC Analyst",
    instructions="You are a QA assistant designed to analyze SEC filings.",
    tools=[lyft_tool],
    verbose=True,
    run_retrieve_sleep_time=1.0,
)

response = agent.chat(
    "Tell me about Lyft's risk factors, as well as response to COVID-19"
)

print(str(response))

