#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/query_engine/citation_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # CitationQueryEngine
# 
# This notebook walks through how to use the CitationQueryEngine
# 
# The CitationQueryEngine can be used with any existing index.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## Setup

import os
from llama_index.llms import OpenAI
from llama_index.query_engine import CitationQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
)

service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0)
)

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

if not os.path.exists("./citation"):
    documents = SimpleDirectoryReader("./data/paul_graham").load_data()
    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context
    )
    index.storage_context.persist(persist_dir="./citation")
else:
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./citation"),
        service_context=service_context,
    )

# ## Create the CitationQueryEngine w/ Default Arguments

query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    # here we can control how granular citation sources are, the default is 512
    citation_chunk_size=512,
)

response = query_engine.query("What did the author do growing up?")

print(response)

# source nodes are 6, because the original chunks of 1024-sized nodes were broken into more granular nodes
print(len(response.source_nodes))

# ##
# Sources start counting at 1, but python arrays start counting at zero!
# 
# Let's confirm the source makes sense.

print(response.source_nodes[0].node.get_text())

print(response.source_nodes[1].node.get_text())

# ## Adjusting Settings
# 
# Note that setting the chunk size larger than the original chunk size of the nodes will have no effect.
# 
# The default node chunk size is 1024, so here, we are not making our citation nodes any more granular.

query_engine = CitationQueryEngine.from_args(
    index,
    # increase the citation chunk size!
    citation_chunk_size=1024,
    similarity_top_k=3,
)

response = query_engine.query("What did the author do growing up?")

print(response)

# should be less source nodes now!
print(len(response.source_nodes))

# ##
# Sources start counting at 1, but python arrays start counting at zero!
# 
# Let's confirm the source makes sense.

print(response.source_nodes[0].node.get_text())

