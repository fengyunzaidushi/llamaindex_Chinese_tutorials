#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/query_engine/recursive_retriever_agents.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Recursive Retriever + Document Agents
# 
# This guide shows how to combine recursive retrieval and "document agents" for advanced decision making over heterogeneous documents.
# 
# There are two motivating factors that lead to solutions for better retrieval:
# - Decoupling retrieval embeddings from chunk-based synthesis. Oftentimes fetching documents by their summaries will return more relevant context to queries rather than raw chunks. This is something that recursive retrieval directly allows.
# - Within a document, users may need to dynamically perform tasks beyond fact-based question-answering. We introduce the concept of "document agents" - agents that have access to both vector search and summary tools for a given document.

# ### Setup and Download Data
# 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.schema import IndexNode
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.llms import OpenAI

wiki_titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]

from pathlib import Path

import requests

for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

# Load all wiki documents
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()

# Define LLM + Service Context + Callback Manager

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

# ## Build Document Agent for each Document
# 

# 
# First we define both a vector index (for semantic search) and summary index (for summarization) for each document. The two query engines are then converted into tools that are passed to an OpenAI function calling agent.
# 
# This document agent can dynamically choose to perform semantic search or summarization within a given document.
# 
# We create a separate document agent for each city.

from llama_index.agent import OpenAIAgent

# Build agents dictionary
agents = {}

for wiki_title in wiki_titles:
    # build vector index
    vector_index = VectorStoreIndex.from_documents(
        city_docs[wiki_title], service_context=service_context
    )
    # build summary index
    summary_index = SummaryIndex.from_documents(
        city_docs[wiki_title], service_context=service_context
    )
    # define query engines
    vector_query_engine = vector_index.as_query_engine()
    list_query_engine = summary_index.as_query_engine()

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    "Useful for summarization questions related to"
                    f" {wiki_title}"
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=list_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(
                    f"Useful for retrieving specific context from {wiki_title}"
                ),
            ),
        ),
    ]

    # build agent
    function_llm = OpenAI(model="gpt-3.5-turbo-0613")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
    )

    agents[wiki_title] = agent

# ## Build Recursive Retriever over these Agents
# 
# Now we define a set of summary nodes, where each node links to the corresponding Wikipedia city article. We then define a `RecursiveRetriever` on top of these Nodes to route queries down to a given node, which will in turn route it to the relevant document agent.
# 
# We finally define a full query engine combining `RecursiveRetriever` into a `RetrieverQueryEngine`.

# define top-level nodes
nodes = []
for wiki_title in wiki_titles:
    # define index node that links to these agents
    wiki_summary = (
        f"This content contains Wikipedia articles about {wiki_title}. Use"
        " this index if you need to lookup specific facts about"
        f" {wiki_title}.\nDo not use this index if you want to analyze"
        " multiple cities."
    )
    node = IndexNode(text=wiki_summary, index_id=wiki_title)
    nodes.append(node)

# define top-level retriever
vector_index = VectorStoreIndex(nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# define recursive retriever
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer

# note: can pass `agents` dict as `query_engine_dict` since every agent can be used as a query engine
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=agents,
    verbose=True,
)

# #### Define Full Query Engine 
# 
# This query engine uses the recursive retriever + response synthesis module to synthesize a response.

response_synthesizer = get_response_synthesizer(
    # service_context=service_context,
    response_mode="compact",
)
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever,
    response_synthesizer=response_synthesizer,
    service_context=service_context,
)

# ## Running Example Queries

# should use Boston agent -> vector tool
response = query_engine.query("Tell me about the sports teams in Boston")

print(response)

# should use Houston agent -> vector tool
response = query_engine.query("Tell me about the sports teams in Houston")

print(response)

# should use Seattle agent -> summary tool
response = query_engine.query(
    "Give me a summary on all the positive aspects of Chicago"
)

print(response)

