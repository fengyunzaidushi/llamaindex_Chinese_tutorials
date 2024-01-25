#!/usr/bin/env python
# coding: utf-8

# # Custom Retriever combining KG Index and VectorStore Index
# 
# Now let's demo how KG Index could be used. We will create a VectorStore Index, KG Index and a Custom Index combining the two.
# 
# Below digrams are showing how in-context learning works:
# 
# ```
#           in-context learning with Llama Index
#                   ┌────┬────┬────┬────┐                  
#                   │ 1  │ 2  │ 3  │ 4  │                  
#                   ├────┴────┴────┴────┤                  
#                   │  Docs/Knowledge   │                  
# ┌───────┐         │        ...        │       ┌─────────┐
# │       │         ├────┬────┬────┬────┤       │         │
# │       │         │ 95 │ 96 │    │    │       │         │
# │       │         └────┴────┴────┴────┘       │         │
# │ User  │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─▶   LLM   │
# │       │                                     │         │
# │       │                                     │         │
# └───────┘    ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  └─────────┘
#     │          ┌──────────────────────────┐        ▲     
#     └────────┼▶│  Tell me ....., please   │├───────┘     
#                └──────────────────────────┘              
#              │ ┌────┐ ┌────┐               │             
#                │ 3  │ │ 96 │                             
#              │ └────┘ └────┘               │             
#               ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ 
# ```
# 
# With VectorStoreIndex, we create embeddings of each node(chunk), and find TopK related ones towards a given question during the query. In the above diagram, nodes `3` and `96` were fetched as the TopK related nodes, used to help answer the user query. 
# 
# With KG Index, we will extract relationships between entities, representing concise facts from each node. It would look something like this:
# 
# ```
# Node Split and Embedding
# 
# ┌────┬────┬────┬────┐
# │ 1  │ 2  │ 3  │ 4  │
# ├────┴────┴────┴────┤
# │  Docs/Knowledge   │
# │        ...        │
# ├────┬────┬────┬────┤
# │ 95 │ 96 │    │    │
# └────┴────┴────┴────┘
# ```
# 
# Then, if we zoomed in of it:
# 
# ```
#        Node Split and Embedding, with Knowledge Graph being extracted
# 
# ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
# │ .─.       .─.    │  .─.       .─.   │            .─.   │  .─.       .─.   │
# │( x )─────▶ y )   │ ( x )─────▶ a )  │           ( j )  │ ( m )◀────( x )  │
# │ `▲'       `─'    │  `─'       `─'   │            `─'   │  `─'       `─'   │
# │  │     1         │        2         │        3    │    │        4         │
# │ .─.              │                  │            .▼.   │                  │
# │( z )─────────────┼──────────────────┼──────────▶( i )─┐│                  │
# │ `◀────┐          │                  │            `─'  ││                  │
# ├───────┼──────────┴──────────────────┴─────────────────┼┴──────────────────┤
# │       │                      Docs/Knowledge           │                   │
# │       │                            ...                │                   │
# │       │                                               │                   │
# ├───────┼──────────┬──────────────────┬─────────────────┼┬──────────────────┤
# │  .─.  └──────.   │  .─.             │                 ││  .─.             │
# │ ( x ◀─────( b )  │ ( x )            │                 └┼▶( n )            │
# │  `─'       `─'   │  `─'             │                  │  `─'             │
# │        95   │    │   │    96        │                  │   │    98        │
# │            .▼.   │  .▼.             │                  │   ▼              │
# │           ( c )  │ ( d )            │                  │  .─.             │
# │            `─'   │  `─'             │                  │ ( x )            │
# └──────────────────┴──────────────────┴──────────────────┴──`─'─────────────┘
# ```
# 
# Where, knowledge, the more granular spliting and information with higher density, optionally multi-hop of `x -> y`, `i -> j -> z -> x` etc... across many more nodes(chunks) than K(in TopK search) could be inlucded in Retrievers. And we believe there are cases that this additional work matters.
# 
# Let's show examples of that now.

# For OpenAI

import os

os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"

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
from llama_index import set_global_service_context

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

import logging
import sys

from IPython.#display import Markdown, #display

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_type = "azure"
openai.api_base = "https://<foo-bar>.openai.azure.com"
openai.api_version = "2022-12-01"
os.environ["OPENAI_API_KEY"] = "youcannottellanyone"
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = AzureOpenAI(
    engine="<foo-bar-deployment>",
    temperature=0,
    openai_api_version=openai.api_version,
    model_kwargs={
        "api_key": openai.api_key,
        "api_base": openai.api_base,
        "api_type": openai.api_type,
        "api_version": openai.api_version,
    },
)

# You need to deploy your own embedding model as well as your own chat completion model
embedding_llm = OpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="<foo-bar-deployment>",
    api_key=openai.api_key,
    api_base=openai.api_base,
    api_type=openai.api_type,
    api_version=openai.api_version,
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_llm,
)

set_global_service_context(service_context)

# ## Prepare for NebulaGraph

get_ipython().run_line_magic('pip', 'install nebula3-python')

os.environ["NEBULA_USER"] = "root"
os.environ["NEBULA_PASSWORD"] = "nebula"
os.environ[
    "NEBULA_ADDRESS"
] = "127.0.0.1:9669"  # assumed we have NebulaGraph 3.5.0 or newer installed locally

# Assume that the graph has already been created
# Create a NebulaGraph cluster with:
# Option 0: `curl -fsSL nebula-up.siwei.io/install.sh | bash`
# Option 1: NebulaGraph Docker Extension https://hub.docker.com/extensions/weygu/nebulagraph-dd-ext
# and that the graph space is called "llamaindex"
# If not, create it with the following commands from NebulaGraph's console:
# CREATE SPACE llamaindex(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);
# :sleep 10;
# USE llamaindex;
# CREATE TAG entity(name string);
# CREATE EDGE relationship(relationship string);
# CREATE TAG INDEX entity_index ON entity(name(256));

space_name = "llamaindex"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg

# ## Load Data from Wikipedia

from llama_index import download_loader

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()

documents = loader.load_data(pages=["2023 in science"], auto_suggest=False)

# ## Create KnowledgeGraphIndex Index

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=10,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)

# ## Create VectorStoreIndex Index

vector_index = VectorStoreIndex.from_documents(documents)

# ## Define a CustomRetriever
# 
# The purpose of this demo was to test the effectiveness of using Knowledge Graph queries for retrieving information that is distributed across multiple nodes in small pieces. To achieve this, we adopted a simple approach: performing retrieval on both sources and then combining them into a single context to be sent to LLM.
# 
# Thanks to the flexible abstraction provided by Llama Index Retriever, implementing this approach was relatively straightforward. We created a new class called `CustomRetriever` which retrieves data from both `VectorIndexRetriever` and `KGTableRetriever`. 

# import QueryBundle
from llama_index import QueryBundle

# import NodeWithScore
from llama_index.schema import NodeWithScore

# Retrievers
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KGTableRetriever,
)

from typing import List

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KGTableRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

# Next, we will create instances of the Vector and KG retrievers, which will be used in the instantiation of the Custom Retriever.

from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine

# create custom retriever
vector_retriever = VectorIndexRetriever(index=vector_index)
kg_retriever = KGTableRetriever(
    index=kg_index, retriever_mode="keyword", include_text=False
)
custom_retriever = CustomRetriever(vector_retriever, kg_retriever)

# create response synthesizer
response_synthesizer = get_response_synthesizer(
    service_context=service_context,
    response_mode="tree_summarize",
)

# ## Create Query Engines
# 
# To enable comparsion, we also create `vector_query_engine`, `kg_keyword_query_engine` together with our `custom_query_engine`.

custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)

vector_query_engine = vector_index.as_query_engine()

kg_keyword_query_engine = kg_index.as_query_engine(
    # setting to false uses the raw triplets instead of adding the text from the corresponding nodes
    include_text=False,
    retriever_mode="keyword",
    response_mode="tree_summarize",
)

# ## Query with different retrievers
# 
# With the above query engines created for corresponding retrievers, let's see how they perform.
# 
# First, we go with the pure knowledge graph.

response = kg_keyword_query_engine.query("Tell me events about NASA")
#display(Markdown(f"<b>{response}</b>"))

# Then the vector store approach.

response = vector_query_engine.query("Tell me events about NASA")
#display(Markdown(f"<b>{response}</b>"))

# Finally, let's do with the one with both vector store and knowledge graph.

response = custom_query_engine.query("Tell me events about NASA")
#display(Markdown(f"<b>{response}</b>"))

# ## Comparison of results
# 
# Let's put results together with their LLM tokens during the query process:
# 
# > Tell me events about NASA.
# 
# |        | VectorStore                                                  | Knowledge Graph + VectorStore                                | Knowledge Graph                                              |
# | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
# | Answer | NASA scientists report evidence for the existence of a second Kuiper Belt, which the New Horizons spacecraft could potentially visit during the late 2020s or early 2030s. NASA is expected to release the first study on UAP in mid-2023. NASA's Venus probe is scheduled to be launched and to arrive on Venus in October, partly to search for signs of life on Venus. NASA is expected to start the Vera Rubin Observatory, the Qitai Radio Telescope, the European Spallation Source and the Jiangmen Underground Neutrino. NASA scientists suggest that a space sunshade could be created by mining the lunar soil and launching it towards the Sun to form a shield against global warming. | NASA announces future space telescope programs on May 21. **NASA publishes images of debris disk on May 23. NASA discovers exoplanet LHS 475 b on May 25.** NASA scientists present evidence for the existence of a second Kuiper Belt on May 29. NASA confirms the start of the next El Niño on June 8. NASA produces the first X-ray of a single atom on May 31. NASA reports the first successful beaming of solar energy from space down to a receiver on the ground on June 1. NASA scientists report evidence that Earth may have formed in just three million years on June 14. NASA scientists report the presence of phosphates on Enceladus, moon of the planet Saturn, on June 14. NASA's Venus probe is scheduled to be launched and to arrive on Venus in October. NASA's MBR Explorer is announced by the United Arab Emirates Space Agency on May 29. NASA's Vera Rubin Observatory is expected to start in 2023. | NASA announced future space telescope programs in mid-2023, published images of a debris disk, and discovered an exoplanet called LHS 475 b. |
# | Cost   | 1897 tokens                                                  | 2046 Tokens                                                  | 159 Tokens                                                   |
# 
# 
# And we could see there are indeed some knowledges added with the help of Knowledge Graph retriever:
# 
# - NASA publishes images of debris disk on May 23.
# - NASA discovers exoplanet LHS 475 b on May 25.
# 
# The additional cost, however, does not seem to be very significant, at `7.28%`: `(2046-1897)/2046`.
# 
# Furthermore, the answer from the knowledge graph is extremely concise (only 159 tokens used!), but is still informative.

# ## Not all cases are advantageous
# 
# While, of course, many other questions do not contain small-grained pieces of knowledges in chunks. In these cases, the extra Knowledge Graph retriever may not that helpful. Let's see this question: "Tell me events about ChatGPT".

response = custom_query_engine.query("Tell me events about ChatGPT")
#display(Markdown(f"<b>{response}</b>"))

response = kg_keyword_query_engine.query("Tell me events about ChatGPT")
#display(Markdown(f"<b>{response}</b>"))

response = vector_query_engine.query("Tell me events about ChatGPT")
#display(Markdown(f"<b>{response}</b>"))

# ## Comparison of results
# 
# We can see that being w/ vs. w/o Knowledge Graph has no unique advantage under this question.
# 
# > Question: Tell me events about ChatGPT.
# 
# |        | VectorStore                                                  | Knowledge Graph + VectorStore                                | Knowledge Graph                                              |
# | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
# | Answer | ChatGPT (released on 30 Nov 2022) is a chatbot and text-generating AI, and a large language model that quickly became highly popular. It is estimated that only two months after its launch, it had 100 million active users. Applications may include solving or supporting school writing assignments, malicious social bots (e.g. for misinformation, propaganda, and scams), and providing inspiration (e.g. for artistic writing or in design or ideation in general). In response to the ChatGPT release, Google released chatbot Bard (21 Mar) with potential for integration into its Web search and, like ChatGPT software, also as a software development helper tool. DuckDuckGo released the DuckAssist feature integrated into its search engine that summarizes information from Wikipedia to answer search queries that are questions (8 Mar). The experimental feature was shut down without explanation on 12 April. Around the time, a proprietary feature by scite.ai was released that delivers answers that use research papers and provide citations for the quoted paper(s). An open letter "Pause Giant AI Experiments" by the Future of Life Institute calls for "AI labs to immediately pause for at least 6 months the training of AI systems more powerful than GPT- | ChatGPT is a chatbot and text-generating AI released on 30 November 2022. It quickly became highly popular, with some estimating that only two months after its launch, it had 100 million active users. Potential applications of ChatGPT include solving or supporting school writing assignments, malicious social bots (e.g. for misinformation, propaganda, and scams), and providing inspiration (e.g. for artistic writing or in design or ideation in general). There was extensive media coverage of views that regard ChatGPT as a potential step towards AGI or sentient machines, also extending to some academic works. Google released chatbot Bard due to effects of the ChatGPT release, with potential for integration into its Web search and, like ChatGPT software, also as a software development helper tool (21 Mar). DuckDuckGo released the DuckAssist feature integrated into its search engine that summarizes information from Wikipedia to answer search queries that are questions (8 Mar). The experimental feature was shut down without explanation on 12 April. Around the same time, a proprietary feature by scite.ai was released that delivers answers that use research papers and provide citations for the quoted paper(s). An open letter "Pause Giant AI Experiments" by the Future of Life | ChatGPT is a language model that outperforms human doctors and has 100 million active users. It was released on 30 November 2022. |
# | Cost   | 1963 Tokens                                                  | 2045 Tokens                                                  | 150 Tokens                                                   |
# 

## create graph
from pyvis.network import Network

g = kg_index.get_networkx_graph(200)
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("2023_Science_Wikipedia_KnowledgeGraph.html")

