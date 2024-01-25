#!/usr/bin/env python
# coding: utf-8

# # LlamaHub Demostration
# 
# Here we give a simple overview of how to use data loaders and tools (for agents) within [LlamaHub](llamahub.ai).
# 
# **NOTES**: 
# 
# - You can learn how to use everything in LlamaHub by clicking into each module and looking at the code snippet.
# - Also, you can find a [full list of notebooks around agent tools here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks).
# - In this guide we'll show how to use `download_loader` and `download_tool`. You can also install `llama-hub` [as a package](https://github.com/run-llama/llama-hub#usage-use-llama-hub-as-pypi-package).
# 

# ## Using a Data Loader
# 

# 
# **NOTE**: for any module on LlamaHub, to use with `download_` functions, note down the class name.

from llama_index.readers import download_loader

SimpleWebPageReader = download_loader("SimpleWebPageReader")

reader = SimpleWebPageReader(html_to_text=True)

docs = reader.load_data(urls=["https://eugeneyan.com/writing/llm-patterns/"])

print(docs[0].get_content()[:400])

# Now you can plug these docs into your downstream LlamaIndex pipeline.

from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()

response = query_engine.query("What are ways to evaluate LLMs?")
print(str(response))

# ## Using an Agent Tool Spec
# 

from llama_index.tools import download_tool

GmailToolSpec = download_tool("GmailToolSpec", refresh_cache=True)

tool_spec = GmailToolSpec()

# plug into your agent
from llama_index.agent import OpenAIAgent

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("What is my most recent email")

