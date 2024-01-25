#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/query_engine/knowledge_graph_rag_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Knowledge Graph RAG Query Engine
# 
# 
# ## Graph RAG
# 
# Graph RAG is an Knowledge-enabled RAG approach to retrieve information from Knowledge Graph on given task. Typically, this is to build context based on entities' SubGraph related to the task.
# 
# ## GraphStore backed RAG vs VectorStore RAG
# 
# As we compared how Graph RAG helps in some use cases in [this tutorial](https://gpt-index.readthedocs.io/en/latest/examples/index_structs/knowledge_graph/KnowledgeGraphIndex_vs_VectorStoreIndex_vs_CustomIndex_combined.html#id1), it's shown Knowledge Graph as the unique format of information could mitigate several issues caused by the nature of the "split and embedding" RAG approach.
# 
# ## Why Knowledge Graph RAG Query Engine
# 

# 
# - Build Knowledge Graph from documents with Llama Index, with LLM or even [local models](https://colab.research.google.com/drive/1G6pcR0pXvSkdMQlAK_P-IrYgo-_staxd?usp=sharing), to do this, we should go for `KnowledgeGraphIndex`.
# - Leveraging existing Knowledge Graph, in this case, we should use `KnowledgeGraphRAGQueryEngine`.
# 
# > Note, the third query engine that's related to KG in Llama Index is `NL2GraphQuery` or `Text2Cypher`, for either exiting KG or not, it could be done with `KnowledgeGraphQueryEngine`.

# Before we start the `Knowledge Graph RAG QueryEngine` demo, let's first get ready for basic preparation of Llama Index.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
# 

#('pip install llama-index')

# For OpenAI

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output

from llama_index import (
    KnowledgeGraphIndex,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
from llama_index.llms import OpenAI

from IPython.#display import Markdown, #display

# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit errors
llm = OpenAI(temperature=0, model="text-davinci-002")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=512)

# For Azure OpenAI
import os
import json
import openai
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    ServiceContext,
)

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
from llama_index.llms import LangChainLLM

import logging
import sys

from IPython.#display import Markdown, #display

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_type = "azure"
openai.api_base = "INSERT AZURE API BASE"
openai.api_version = "2023-05-15"
os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = AzureOpenAI(
    engine="INSERT DEPLOYMENT NAME",
    temperature=0,
    model="gpt-35-turbo",
)

# You need to deploy your own embedding model as well as your own chat completion model
embedding_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="INSERT DEPLOYMENT NAME",
    api_key=openai.api_key,
    api_base=openai.api_base,
    api_type=openai.api_type,
    api_version=openai.api_version,
)
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_model,
)

# ## Prepare for NebulaGraph
# 
# We take [NebulaGraphStore](https://gpt-index.readthedocs.io/en/stable/examples/index_structs/knowledge_graph/NebulaGraphKGIndexDemo.html) as an example in this demo, thus before next step to perform Graph RAG on existing KG, let's ensure we have a running NebulaGraph with defined data schema.
# 
# This step installs the clients of NebulaGraph, and prepare contexts that defines a [NebulaGraph Graph Space](https://docs.nebula-graph.io/3.6.0/1.introduction/2.data-model/).

# Create a NebulaGraph (version 3.5.0 or newer) cluster with:
# Option 0 for machines with Docker: `curl -fsSL nebula-up.siwei.io/install.sh | bash`
# Option 1 for Desktop: NebulaGraph Docker Extension https://hub.docker.com/extensions/weygu/nebulagraph-dd-ext

# If not, create it with the following commands from NebulaGraph's console:
# CREATE SPACE llamaindex(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);
# :sleep 10;
# USE llamaindex;
# CREATE TAG entity(name string);
# CREATE EDGE relationship(relationship string);
# :sleep 10;
# CREATE TAG INDEX entity_index ON entity(name(256));

get_ipython().run_line_magic('pip', 'install ipython-ngql nebula3-python')

os.environ["NEBULA_USER"] = "root"
os.environ["NEBULA_PASSWORD"] = "nebula"  # default is "nebula"
os.environ[
    "NEBULA_ADDRESS"
] = "127.0.0.1:9669"  # assumed we have NebulaGraph installed locally

space_name = "llamaindex"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg

# Then we could instiatate a `NebulaGraphStore`, in order to create a `StorageContext`'s `graph_store` as it.

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Here, we assumed to have the same Knowledge Graph from [this turtorial](https://gpt-index.readthedocs.io/en/latest/examples/query_engine/knowledge_graph_query_engine.html#optional-build-the-knowledge-graph-with-llamaindex)

# ## Perform Graph RAG Query
# 
# Finally, let's demo how to do Graph RAG towards an existing Knowledge Graph.
# 
# All we need to do is to use `RetrieverQueryEngine` and configure the retriver of it to be `KnowledgeGraphRAGRetriever`.
# 
# The `KnowledgeGraphRAGRetriever` performs the following steps:
# 
# - Search related Entities of the quesion/task
# - Get SubGraph of those Entities (default 2-depth) from the KG
# - Build Context based on the SubGraph
# 
# Please note, the way to Search related Entities could be either Keyword extraction based or Embedding based, which is controlled by argument `retriever_mode` of the `KnowledgeGraphRAGRetriever`, and supported options are:
# - "keyword"
# - "embedding"(not yet implemented)
# - "keyword_embedding"(not yet implemented)
# 
# Here is the example on how to use `RetrieverQueryEngine` and `KnowledgeGraphRAGRetriever`:

from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import KnowledgeGraphRAGRetriever

graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(
    graph_rag_retriever, service_context=service_context
)

# Then we can query it like:

response = query_engine.query(
    "Tell me about Peter Quill?",
)
#display(Markdown(f"<b>{response}</b>"))

response = await query_engine.aquery(
    "Tell me about Peter Quill?",
)
#display(Markdown(f"<b>{response}</b>"))

# #
# 
# The nature of (Sub)Graph RAG and nl2graphquery are different. No one is better than the other but just when one fits more in certain type of questions. To understand more on how they differ from the other, see [this demo](https://www.siwei.io/en/demos/graph-rag/) comparing the two.
# 
# <video width="938" height="800" 
#        src="https://github.com/siwei-io/talks/assets/1651790/05d01e53-d819-4f43-9bf1-75549f7f2be9"  
#        controls>
# </video>
# 
# While in real world cases, we may not always know which approach works better, thus, one way to best leverage KG in RAG are fetching both retrieval results as context and letting LLM + Prompt generate answer with them all being involved.
# 
# So, optionally, we could choose to synthesise answer from two piece of retrieved context from KG:
# - Graph RAG, the default retrieval method, which extracts subgraph that's related to the key entities in the question.
# - NL2GraphQuery, generate Knowledge Graph Query based on query and the Schema of the Knowledge Graph, which is by default switched off.
# 
# We could set `with_nl2graphquery=True` to enable it like:

graph_rag_retriever_with_nl2graphquery = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
    with_nl2graphquery=True,
)

query_engine_with_nl2graphquery = RetrieverQueryEngine.from_args(
    graph_rag_retriever_with_nl2graphquery, service_context=service_context
)

response = query_engine_with_nl2graphquery.query(
    "What do you know about Peter Quill?",
)
#display(Markdown(f"<b>{response}</b>"))

# And let's check the response's metadata to know more details of the retrival of Graph RAG with nl2graphquery by inspecting `response.metadata`.
# 
# - **text2Cypher**, it generates a Cypher Query towards the answer as the context.
# 
# ```cypher
# Graph Store Query: MATCH (e:`entity`)-[r:`relationship`]->(e2:`entity`)
# WHERE e.`entity`.`name` == 'Peter Quill'
# RETURN e2.`entity`.`name`
# ```
# - **SubGraph RAG**, it get the SubGraph of 'Peter Quill' to build the context.
# 
# - Finally, it combined the two nodes of context, to synthesize the answer.

import pprint

pp = pprint.PrettyPrinter()
pp.pprint(response.metadata)

