#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/CognitiveSearchIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Azure Cognitive Search

# ## Basic Example
# 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys
from IPython.#display import Markdown, #display

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# logger = logging.getLogger(__name__)

#!{sys.executable} -m pip install llama-index
#!{sys.executable} -m pip install azure-search-documents==11.4.0b8
#!{sys.executable} -m pip install azure-identity

# set up OpenAI
import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# set up Azure Cognitive Search
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

search_service_name = getpass.getpass("Azure Cognitive Search Service Name")

key = getpass.getpass("Azure Cognitive Search Key")

cognitive_search_credential = AzureKeyCredential(key)

service_endpoint = f"https://{search_service_name}.search.windows.net"

index_name = "quickstart"

# Use index client to demonstrate creating an index
index_client = SearchIndexClient(
    endpoint=service_endpoint,
    credential=cognitive_search_credential,
)

# Use search client to demonstration using existing index
search_client = SearchClient(
    endpoint=service_endpoint,
    index_name=index_name,
    credential=cognitive_search_credential,
)

# ## Create Index (if it does not exist)

# Demonstrates creating a vector index named quickstart01 if one doesn't exist. The index has the following fields:
# - id (Edm.String)
# - content (Edm.String)
# - embedding (Edm.SingleCollection)
# - li_jsonMetadata (Edm.String)
# - li_doc_id (Edm.String)
# - author (Edm.String)
# - theme (Edm.String)
# - director (Edm.String)

from azure.search.documents import SearchClient
from llama_index.vector_stores import CognitiveSearchVectorStore
from llama_index.vector_stores.cogsearch import (
    IndexManagement,
    MetadataIndexFieldType,
    CognitiveSearchVectorStore,
)

# Example of a complex mapping, metadata field 'theme' is mapped to a differently name index field 'topic' with its type explicitly set
metadata_fields = {
    "author": "author",
    "theme": ("topic", MetadataIndexFieldType.STRING),
    "director": "director",
}

# A simplified metadata specification is available if all metadata and index fields are similarly named
# metadata_fields = {"author", "theme", "director"}

vector_store = CognitiveSearchVectorStore(
    search_or_index_client=index_client,
    index_name=index_name,
    filterable_metadata_field_keys=metadata_fields,
    index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
    id_field_key="id",
    chunk_field_key="content",
    embedding_field_key="embedding",
    metadata_string_field_key="li_jsonMetadata",
    doc_id_field_key="li_doc_id",
)

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# define embedding function
from llama_index.embeddings import OpenAIEmbedding
from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    VectorStoreIndex,
)

embed_model = OpenAIEmbedding()

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)

# Query Data
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What did the author do growing up?")
#display(Markdown(f"<b>{response}</b>"))

response = query_engine.query(
    "What did the author learn?",
)
#display(Markdown(f"<b>{response}</b>"))

# ## Use Existing Index

from llama_index.vector_stores import CognitiveSearchVectorStore
from llama_index.vector_stores.cogsearch import (
    IndexManagement,
    MetadataIndexFieldType,
    CognitiveSearchVectorStore,
)

index_name = "quickstart"

metadata_fields = {
    "author": "author",
    "theme": ("topic", MetadataIndexFieldType.STRING),
    "director": "director",
}
vector_store = CognitiveSearchVectorStore(
    search_or_index_client=search_client,
    filterable_metadata_field_keys=metadata_fields,
    index_management=IndexManagement.NO_VALIDATION,
    id_field_key="id",
    chunk_field_key="content",
    embedding_field_key="embedding",
    metadata_string_field_key="li_jsonMetadata",
    doc_id_field_key="li_doc_id",
)

# define embedding function
from llama_index.embeddings import OpenAIEmbedding
from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    VectorStoreIndex,
)

embed_model = OpenAIEmbedding()

storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = VectorStoreIndex.from_documents(
    [], storage_context=storage_context, service_context=service_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What was a hard moment for the author?")
#display(Markdown(f"<b>{response}</b>"))

response = query_engine.query("Who is the author?")
#display(Markdown(f"<b>{response}</b>"))

import time

query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("What happened at interleaf?")

start_time = time.time()

token_count = 0
for token in response.response_gen:
    print(token, end="")
    token_count += 1

time_elapsed = time.time() - start_time
tokens_per_second = token_count / time_elapsed

print(f"\n\nStreamed output at {tokens_per_second} tokens/s")

# ## Adding a document to existing index

response = query_engine.query("What colour is the sky?")
#display(Markdown(f"<b>{response}</b>"))

from llama_index import Document

index.insert_nodes([Document(text="The sky is indigo today")])

response = query_engine.query("What colour is the sky?")
#display(Markdown(f"<b>{response}</b>"))

# ## Filtering

from llama_index.schema import TextNode

nodes = [
    TextNode(
        text="The Shawshank Redemption",
        metadata={
            "author": "Stephen King",
            "theme": "Friendship",
        },
    ),
    TextNode(
        text="The Godfather",
        metadata={
            "director": "Francis Ford Coppola",
            "theme": "Mafia",
        },
    ),
    TextNode(
        text="Inception",
        metadata={
            "director": "Christopher Nolan",
        },
    ),
]

index.insert_nodes(nodes)

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What is inception about?")

