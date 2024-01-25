#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/chroma_metadata_filter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Chroma Vector Store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# #### Creating a Chroma Index

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import os
import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
import openai

openai.api_key = "sk-"

import chromadb

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

from llama_index import VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore
from IPython.#display import Markdown, #display

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

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(nodes, storage_context=storage_context)

# ## One Exact Match Filter

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

# ## Multiple Exact Match Metadata Filters

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="theme", value="Mafia"),
        MetadataFilter(key="year", value=1972),
    ]
)

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What is inception about?")

# ## Multiple Metadata Filters with `AND` condition

from llama_index.vector_stores.types import (
    FilterOperator,
    FilterCondition,
)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="theme", value="Fiction"),
        MetadataFilter(key="year", value=1997, operator=FilterOperator.GT),
    ],
    condition=FilterCondition.AND,
)

retriever = index.as_retriever(filters=filters)
retriever.retrieve("Harry Potter?")

# ## Multiple Metadata Filters with `OR` condition

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

