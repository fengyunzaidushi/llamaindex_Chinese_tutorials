#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/pinecone_metadata_filter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Qdrant Vector Store - Metadata Filter

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index qdrant_client')

# Build a Pinecone Index and connect to it

import logging
import sys
import os

import qdrant_client
from IPython.#display import Markdown, #display
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

client = qdrant_client.QdrantClient(
    # you can use :memory: mode for fast and light-weight experiments,
    # it does not require to have Qdrant deployed anywhere
    # but requires qdrant-client >= 1.1.1
    location=":memory:"
    # otherwise set Qdrant instance address with:
    # uri="http://<host>:<port>"
    # set API KEY for Qdrant Cloud
    # api_key="<qdrant-api-key>",
)

# Build the PineconeVectorStore and VectorStoreIndex

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

import openai

from llama_index.storage.storage_context import StorageContext

openai.api_key = "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"

vector_store = QdrantVectorStore(
    client=client, collection_name="test_collection_1"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

# Define metadata filters

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

# Retrieve from vector store with filters

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What is inception about?")

# Multiple Metadata Filters with `AND` condition

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

# Use keyword arguments specific to pinecone

retriever = index.as_retriever(
    vector_store_kwargs={"filter": {"theme": "Mafia"}}
)
retriever.retrieve("What is inception about?")

