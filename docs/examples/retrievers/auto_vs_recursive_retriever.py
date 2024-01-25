#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/retrievers/auto_vs_recursive_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Comparing Methods for Structured Retrieval (Auto-Retrieval vs. Recursive Retrieval)
# 

# 
# This can fail if the set of documents is large - it can be hard to disambiguate raw chunks, and you're not guaranteed to filter for the set of documents that contain relevant context.
# 

# 
# - **Metadata Filters + Auto-Retrieval**: Tag each document with the right set of metadata. During query-time, use auto-retrieval to infer metadata filters along with passing through the query string for semantic search.
# - **Store Document Hierarchies (summaries -> raw chunks) + Recursive Retrieval**: Embed document summaries and map that to the set of raw chunks for each document. During query-time, do recursive retrieval to first fetch summaries before fetching documents.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import nest_asyncio

nest_asyncio.apply()

import logging
import sys
from llama_index import SimpleDirectoryReader, SummaryIndex, ServiceContext

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

wiki_titles = ["Michael Jordan", "Elon Musk", "Richard Branson", "Rihanna"]
wiki_metadatas = {
    "Michael Jordan": {
        "category": "Sports",
        "country": "United States",
    },
    "Elon Musk": {
        "category": "Business",
        "country": "United States",
    },
    "Richard Branson": {
        "category": "Business",
        "country": "UK",
    },
    "Rihanna": {
        "category": "Music",
        "country": "Barbados",
    },
}

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
docs_dict = {}
for wiki_title in wiki_titles:
    doc = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()[0]

    doc.metadata.update(wiki_metadatas[wiki_title])
    docs_dict[wiki_title] = doc

from llama_index.llms import OpenAI
from llama_index.callbacks import LlamaDebugHandler, CallbackManager

llm = OpenAI("gpt-4")
callback_manager = CallbackManager([LlamaDebugHandler()])
service_context = ServiceContext.from_defaults(
    llm=llm, callback_manager=callback_manager, chunk_size=256
)

# ## Metadata Filters + Auto-Retrieval
# 

# 
# During retrieval-time, we then perform "auto-retrieval" to infer the relevant set of metadata filters.

## Setup Weaviate
import weaviate

# cloud
resource_owner_config = weaviate.AuthClientPassword(
    username="username",
    password="password",
)
client = weaviate.Client(
    "https://llamaindex-test-ul4sgpxc.weaviate.network",
    auth_client_secret=resource_owner_config,
)

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import WeaviateVectorStore
from IPython.#display import Markdown, #display

# drop items from collection first
client.schema.delete_class("LlamaIndex")

from llama_index.storage.storage_context import StorageContext

# If you want to load the index later, be sure to give it a name!
vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="LlamaIndex"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# NOTE: you may also choose to define a index_name manually.
# index_name = "test_prefix"
# vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)

# validate that the schema was created
class_schema = client.schema.get("LlamaIndex")
#display(class_schema)

index = VectorStoreIndex(
    [], storage_context=storage_context, service_context=service_context
)

# add documents to index
for wiki_title in wiki_titles:
    index.insert(docs_dict[wiki_title])

from llama_index.indices.vector_store.retrievers import (
    VectorIndexAutoRetriever,
)
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo

vector_store_info = VectorStoreInfo(
    content_info="brief biography of celebrities",
    metadata_info=[
        MetadataInfo(
            name="category",
            type="str",
            description=(
                "Category of the celebrity, one of [Sports, Entertainment,"
                " Business, Music]"
            ),
        ),
        MetadataInfo(
            name="country",
            type="str",
            description=(
                "Country of the celebrity, one of [United States, Barbados,"
                " Portugal]"
            ),
        ),
    ],
)
retriever = VectorIndexAutoRetriever(
    index,
    vector_store_info=vector_store_info,
    service_context=service_context,
    max_top_k=10000,
)

# NOTE: the "set top-k to 10000" is a hack to return all data.
# Right now auto-retrieval will always return a fixed top-k, there's a TODO to allow it to be None
# to fetch all data.
# So it's theoretically possible to have the LLM infer a None top-k value.
nodes = retriever.retrieve(
    "Tell me about a celebrity from the United States, set top k to 10000"
)

print(f"Number of nodes: {len(nodes)}")
for node in nodes:
    print(node.node.get_content())

nodes = retriever.retrieve(
    "Tell me about the childhood of a popular sports celebrity in the United"
    " States"
)
for node in nodes:
    print(node.node.get_content())

nodes = retriever.retrieve(
    "Tell me about the college life of a billionaire who started at company at"
    " the age of 16"
)
for node in nodes:
    print(node.node.get_content())

nodes = retriever.retrieve("Tell me about the childhood of a UK billionaire")
for node in nodes:
    print(node.node.get_content())

# ## Build Recursive Retriever over Document Summaries

from llama_index.schema import IndexNode

# define top-level nodes and vector retrievers
nodes = []
vector_query_engines = {}
vector_retrievers = {}

for wiki_title in wiki_titles:
    # build vector index
    vector_index = VectorStoreIndex.from_documents(
        [docs_dict[wiki_title]], service_context=service_context
    )
    # define query engines
    vector_query_engine = vector_index.as_query_engine()
    vector_query_engines[wiki_title] = vector_query_engine
    vector_retrievers[wiki_title] = vector_index.as_retriever()

    # save summaries
    out_path = Path("summaries") / f"{wiki_title}.txt"
    if not out_path.exists():
        # use LLM-generated summary
        summary_index = SummaryIndex.from_documents(
            [docs_dict[wiki_title]], service_context=service_context
        )

        summarizer = summary_index.as_query_engine(
            response_mode="tree_summarize"
        )
        response = await summarizer.aquery(
            f"Give me a summary of {wiki_title}"
        )

        wiki_summary = response.response
        Path("summaries").mkdir(exist_ok=True)
        with open(out_path, "w") as fp:
            fp.write(wiki_summary)
    else:
        with open(out_path, "r") as fp:
            wiki_summary = fp.read()

    print(f"**Summary for {wiki_title}: {wiki_summary}")
    node = IndexNode(text=wiki_summary, index_id=wiki_title)
    nodes.append(node)

# define top-level retriever
top_vector_index = VectorStoreIndex(nodes)
top_vector_retriever = top_vector_index.as_retriever(similarity_top_k=1)

# define recursive retriever
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer

# note: can pass `agents` dict as `query_engine_dict` since every agent can be used as a query engine
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": top_vector_retriever, **vector_retrievers},
    # query_engine_dict=vector_query_engines,
    verbose=True,
)

# ?
nodes = recursive_retriever.retrieve(
    "Tell me about a celebrity from the United States"
)
for node in nodes:
    print(node.node.get_content())

nodes = recursive_retriever.retrieve(
    "Tell me about the childhood of a billionaire who started at company at"
    " the age of 16"
)
for node in nodes:
    print(node.node.get_content())

