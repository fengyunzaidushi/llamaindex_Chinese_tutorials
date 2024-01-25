#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/query_engine/knowledge_graph_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Knowledge Graph Query Engine
# 
# Creating a Knowledge Graph usually involves specialized and complex tasks. However, by utilizing the Llama Index (LLM), the KnowledgeGraphIndex, and the GraphStore, we can facilitate the creation of a relatively effective Knowledge Graph from any data source supported by [Llama Hub](https://llamahub.ai/).
# 
# Furthermore, querying a Knowledge Graph often requires domain-specific knowledge related to the storage system, such as Cypher. But, with the assistance of the LLM and the LlamaIndex KnowledgeGraphQueryEngine, this can be accomplished using Natural Language!
# 

# 
# - Extract and Set Up a Knowledge Graph using the Llama Index
# - Query a Knowledge Graph using Cypher
# - Query a Knowledge Graph using Natural Language

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# Let's first get ready for basic preparation of Llama Index.

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
openai.api_version = "2022-12-01"
os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

lc_llm = AzureOpenAI(
    deployment_name="INSERT DEPLOYMENT NAME",
    temperature=0,
    openai_api_version=openai.api_version,
    model_kwargs={
        "api_key": openai.api_key,
        "api_base": openai.api_base,
        "api_type": openai.api_type,
        "api_version": openai.api_version,
    },
)
llm = LangChainLLM(lc_llm)

# You need to deploy your own embedding model as well as your own chat completion model
embedding_llm = OpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="INSERT DEPLOYMENT NAME",
    api_key=openai.api_key,
    api_base=openai.api_base,
    api_type=openai.api_type,
    api_version=openai.api_version,
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_llm,
)

# ## Prepare for NebulaGraph
# 
# Before next step to creating the Knowledge Graph, let's ensure we have a running NebulaGraph with defined data schema.

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

# Prepare for StorageContext with graph_store as NebulaGraphStore

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# ## (Optional)Build the Knowledge Graph with LlamaIndex
# 
# With the help of Llama Index and LLM defined, we could build Knowledge Graph from given documents.
# 
# If we have a Knowledge Graph on NebulaGraphStore already, this step could be skipped

# ### Step 1, load data from Wikipedia for "Guardians of the Galaxy Vol. 3"

from llama_index import download_loader

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()

documents = loader.load_data(
    pages=["Guardians of the Galaxy Vol. 3"], auto_suggest=False
)

# ### Step 2, Generate a KnowledgeGraphIndex with NebulaGraph as graph_store
# 
# Then, we will create a KnowledgeGraphIndex to enable Graph based RAG, see [here](https://gpt-index.readthedocs.io/en/latest/examples/index_structs/knowledge_graph/KnowledgeGraphIndex_vs_VectorStoreIndex_vs_CustomIndex_combined.html) for deails, apart from that, we have a Knowledge Graph up and running for other purposes, too!

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=10,
    service_context=service_context,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)

# Now we have a Knowledge Graph on NebulaGraph cluster under space named `llamaindex` about the 'Guardians of the Galaxy Vol. 3' movie, let's play with it a little bit.

# install related packages, password is nebula by default
get_ipython().run_line_magic('pip', 'install ipython-ngql networkx pyvis')
get_ipython().run_line_magic('load_ext', 'ngql')
get_ipython().run_line_magic('ngql', '--address 127.0.0.1 --port 9669 --user root --password <password>')

# Query some random Relationships with Cypher
get_ipython().run_line_magic('ngql', 'USE llamaindex;')
get_ipython().run_line_magic('ngql', 'MATCH ()-[e]->() RETURN e LIMIT 10')

# draw the result

get_ipython().run_line_magic('ng_draw', '')

# ## Asking the Knowledge Graph
# 
# Finally, let's demo how to Query Knowledge Graph with Natural language!
# 
# Here, we will leverage the `KnowledgeGraphQueryEngine`, with `NebulaGraphStore` as the `storage_context.graph_store`.

from llama_index.query_engine import KnowledgeGraphQueryEngine

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

query_engine = KnowledgeGraphQueryEngine(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
)

response = query_engine.query(
    "Tell me about Peter Quill?",
)
#display(Markdown(f"<b>{response}</b>"))

graph_query = query_engine.generate_query(
    "Tell me about Peter Quill?",
)

graph_query = graph_query.replace("WHERE", "\n  WHERE").replace(
    "RETURN", "\nRETURN"
)

#display(
    Markdown(
        f"""
```cypher
{graph_query}
```
"""
    )
)

# We could see it helps generate the Graph query:
# 
# ```cypher
# MATCH (p:`entity`)-[:relationship]->(e:`entity`) 
#   WHERE p.`entity`.`name` == 'Peter Quill' 
# RETURN e.`entity`.`name`;
# ```
# And synthese the question based on its result:
# 
# ```json
# {'e2.entity.name': ['grandfather', 'alternate version of Gamora', 'Guardians of the Galaxy']}
# ```

# Of course we still could query it, too! And this query engine could be our best Graph Query Language learning bot, then :).

get_ipython().run_cell_magic('ngql', '', "MATCH (p:`entity`)-[e:relationship]->(m:`entity`)\n  WHERE p.`entity`.`name` == 'Peter Quill'\nRETURN p.`entity`.`name`, e.relationship, m.`entity`.`name`;\n")

# And change the query to be rendered

get_ipython().run_cell_magic('ngql', '', "MATCH (p:`entity`)-[e:relationship]->(m:`entity`)\n  WHERE p.`entity`.`name` == 'Peter Quill'\nRETURN p, e, m;\n")

get_ipython().run_line_magic('ng_draw', '')

# The results of this knowledge-fetching query could not be more clear from the renderred graph then.
