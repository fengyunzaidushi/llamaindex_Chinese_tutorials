#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/tools/OnDemandLoaderTool.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # OnDemandLoaderTool Tutorial
# 
# Our `OnDemandLoaderTool` is a powerful agent tool that allows for "on-demand" data querying from any data source on LlamaHub.
# 
# This tool takes in a `BaseReader` data loader, and when called will 1) load data, 2) index data, and 3) query the data.
# 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.tools.ondemand_loader_tool import OnDemandLoaderTool
from llama_index.readers.wikipedia import WikipediaReader
from typing import List

from pydantic import BaseModel

# ### Define Tool
# 
# We first define the `WikipediaReader`. Note that the `load_data` interface to `WikipediaReader` takes in a list of `pages`. By default, this queries the Wikipedia search endpoint which will autosuggest the relevant pages.
# 
# We then wrap it into our `OnDemandLoaderTool`.
# 
# By default since we don't specify the `index_cls`, a simple vector store index is initialized.

reader = WikipediaReader()

tool = OnDemandLoaderTool.from_defaults(
    reader,
    name="Wikipedia Tool",
    description="A tool for loading and querying articles from Wikipedia",
)

# #### Testing
# 
# We can try running the tool by itself (or as a LangChain tool), just to showcase what the interface is like! 
# 
# Note that besides the arguments required for the data loader, the tool also takes in a `query_str` which will be
# the query against the index.

# run tool by itself
tool(["Berlin"], query_str="What's the arts and culture scene in Berlin?")

# run tool as langchain structured tool
lc_tool = tool.to_langchain_structured_tool(verbose=True)

lc_tool.run(
    tool_input={
        "pages": ["Berlin"],
        "query_str": "What's the arts and culture scene in Berlin?",
    }
)

# ##
# 
# For tutorial purposes, the agent just has access to one tool - the Wikipedia Reader
# 
# Note that we need to use Structured Tools from LangChain.

from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)

agent = initialize_agent(
    [lc_tool],
    llm=llm,
    agent="structured-chat-zero-shot-react-description",
    verbose=True,
)

# # Now let's run some queries! 
# 
# The OnDemandLoaderTool allows the agent to simultaneously 1) load the data from Wikipedia, 2) query that data.

agent.run("Tell me about the arts and culture of Berlin")

agent.run("Tell me about the critical reception to The Departed")

