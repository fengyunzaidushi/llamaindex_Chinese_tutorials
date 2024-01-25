#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/WeaviateIndex_metadata_filter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Weaviate Vector Store Metadata Filter

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index weaviate-client')

# #### Creating a Weaviate Client

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-<your key here>"
openai.api_key = os.environ["OPENAI_API_KEY"]

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import weaviate

# cloud
resource_owner_config = weaviate.AuthClientPassword(
    username="",
    password="",
)
client = weaviate.Client(
    "https://test.weaviate.network",
    auth_client_secret=resource_owner_config,
)

# local
# client = weaviate.Client("http://localhost:8081")

# #### Load documents, build the VectorStoreIndex

from llama_index import VectorStoreIndex
from llama_index.vector_stores import WeaviateVectorStore
from IPython.#display import Markdown, #display

# ## Metadata Filtering
# 
# Let's insert a dummy document, and try to filter so that only that document is returned.

from llama_index.schema import TextNode

nodes = [
    TextNode(
        text="The Shawshank Redemption",
        metadata={
            "author": "Stephen King",
            "theme": "Friendship",
            "year": 1994,
        },
    ),
    TextNode(
        text="The Godfather",
        metadata={
            "director": "Francis Ford Coppola",
            "theme": "Mafia",
            "year": 1972,
        },
    ),
    TextNode(
        text="Inception",
        metadata={
            "director": "Christopher Nolan",
            "theme": "Fiction",
            "year": 2010,
        },
    ),
    TextNode(
        text="To Kill a Mockingbird",
        metadata={
            "author": "Harper Lee",
            "theme": "Mafia",
            "year": 1960,
        },
    ),
    TextNode(
        text="1984",
        metadata={
            "author": "George Orwell",
            "theme": "Totalitarianism",
            "year": 1949,
        },
    ),
    TextNode(
        text="The Great Gatsby",
        metadata={
            "author": "F. Scott Fitzgerald",
            "theme": "The American Dream",
            "year": 1925,
        },
    ),
    TextNode(
        text="Harry Potter and the Sorcerer's Stone",
        metadata={
            "author": "J.K. Rowling",
            "theme": "Fiction",
            "year": 1997,
        },
    ),
]

from llama_index.storage.storage_context import StorageContext

vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="LlamaIndex_filter"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

retriever = index.as_retriever()
retriever.retrieve("What is inception?")

from llama_index.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="theme", operator=FilterOperator.EQ, value="Mafia"),
    ]
)

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What is inception about?")

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="theme", value="Mafia"),
        MetadataFilter(key="year", value=1972),
    ]
)

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What is inception?")

from llama_index.vector_stores.types import (
    FilterOperator,
    FilterCondition,
)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="theme", value="Fiction"),
        MetadataFilter(key="year", value=1997, operator=FilterOperator.GT),
    ],
    condition=FilterCondition.OR,
)

retriever = index.as_retriever(filters=filters)
retriever.retrieve("Harry Potter?")

