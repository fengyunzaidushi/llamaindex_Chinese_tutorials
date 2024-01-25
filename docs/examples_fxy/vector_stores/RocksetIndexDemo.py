#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/RocksetIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Rockset Vector Store
# 
# As a real-time search and analytics database, Rockset uses indexing to deliver scalable and performant personalization, product search, semantic search, chatbot applications, and more.
# Since Rockset is purpose-built for real-time, you can build these responsive applications on constantly updating, streaming data. 
# By integrating Rockset with LlamaIndex, you can easily use LLMs on your own real-time data for production-ready vector search applications.
# 
# We'll walk through a demonstration of how to use Rockset as a vector store in LlamaIndex. 
# 
# ## Tutorial

# We'll ingest text from a file and ask questions about the content.
# 
# ### Setting Up Your Environment
# 1. Create a [collection](https://rockset.com/docs/collections) from the Rockset console with the [Write API](https://rockset.com/docs/write-api/) as your source.
# Name your collection `llamaindex_demo`. Configure the following [ingest transformation](https://rockset.com/docs/ingest-transformation) 
# with [`VECTOR_ENFORCE`](https://rockset.com/docs/vector-functions) to define your embeddings field and take advantage of performance and storage optimizations:
# ```sql
# SELECT 
#     _input.* EXCEPT(_meta), 
#     VECTOR_ENFORCE(
#         _input.embedding,
#         1536,
#         'float'
#     ) as embedding
# FROM _input
# ```
# 
# 2. Create an [API key](https://rockset.com/docs/iam) from the Rockset console and set the `ROCKSET_API_KEY` environment variable.
# Find your API server [here](http://rockset.com/docs/rest-api#introduction) and set the `ROCKSET_API_SERVER` environment variable. 
# Set the `OPENAI_API_KEY` environment variable.
# 
# 3. Install the dependencies.
# ```shell
# pip3 install llama_index rockset 
# ```
# 
# 4. LlamaIndex allows you to ingest data from a variety of sources. 
# For this example, we'll read from a text file named `constitution.txt`, which is a transcript of the American Constitution, found [here](https://www.archives.gov/founding-docs/constitution-transcript). 
# 
# ### Data ingestion 
# Use LlamaIndex's `SimpleDirectoryReader` class to convert the text file to a list of `Document` objects.

from llama_index import SimpleDirectoryReader

docs = SimpleDirectoryReader(
    input_files=["{path to}/consitution.txt"]
).load_data()

from llama_index import ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(temperature=0.8, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

from llama_index import StorageContext
from llama_index.vector_stores import RocksetVectorStore

vector_store = RocksetVectorStore(collection="llamaindex_demo")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Add documents to the `llamaindex_demo` collection and create an index.

from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    docs, storage_context=storage_context, service_context=service_context
)

# ### Querying
# Ask a question about your document and generate a response.

response = index.as_query_engine(service_context=service_context).query(
    "What is the duty of the president?"
)

print(str(response))

# 
# Run the program.
# ```text
# $ python3 main.py
# The duty of the president is to faithfully execute the Office of President of the United States, preserve, protect and defend the Constitution of the United States, serve as the Commander in Chief of the Army and Navy, grant reprieves and pardons for offenses against the United States (except in cases of impeachment), make treaties and appoint ambassadors and other public ministers, take care that the laws be faithfully executed, and commission all the officers of the United States.
# ```
# 
# ## Metadata Filtering
# Metadata filtering allows you to retrieve relevant documents that match specific filters.
# 
# 1. Add nodes to your vector store and create an index.

from llama_index.vector_stores import RocksetVectorStore
from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores.types import NodeWithEmbedding
from llama_index.schema import TextNode

nodes = [
    NodeWithEmbedding(
        node=TextNode(
            text="Apples are blue",
            metadata={"type": "fruit"},
        ),
        embedding=[],
    )
]
index = VectorStoreIndex(
    nodes,
    storage_context=StorageContext.from_defaults(
        vector_store=RocksetVectorStore(collection="llamaindex_demo")
    ),
)

# 2. Define metadata filters.

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="type", value="fruit")]
)

# 3. Retrieve relevant documents that satisfy the filters.

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What colors are apples?")

# ## Creating an Index from an Existing Collection
# You can create indices with data from existing collections.

from llama_index import VectorStoreIndex
from llama_index.vector_stores import RocksetVectorStore

vector_store = RocksetVectorStore(collection="llamaindex_demo")

index = VectorStoreIndex.from_vector_store(vector_store)

# ## Creating an Index from a New Collection
# You can also create a new Rockset collection to use as a vector store.

from llama_index.vector_stores import RocksetVectorStore

vector_store = RocksetVectorStore.with_new_collection(
    collection="llamaindex_demo",  # name of new collection
    dimensions=1536,  # specifies length of vectors in ingest tranformation (optional)
    # other RocksetVectorStore args
)

index = VectorStoreIndex(
    nodes,
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
)

# ## Configuration
# * **collection**: Name of the collection to query (required).
# 
# ```python
# RocksetVectorStore(collection="my_collection")
# ```
# 
# * **workspace**: Name of the workspace containing the collection. Defaults to `"commons"`.
# ```python
# RocksetVectorStore(worksapce="my_workspace")
# ```
# 
# * **api_key**: The API key to use to authenticate Rockset requests. Ignored if `client` is passed in. Defaults to the `ROCKSET_API_KEY` environment variable.
# ```python
# RocksetVectorStore(api_key="<my key>")
# ```
# 
# * **api_server**: The API server to use for Rockset requests. Ignored if `client` is passed in. Defaults to the `ROCKSET_API_KEY` environment variable or `"https://api.use1a1.rockset.com"` if the `ROCKSET_API_SERVER` is not set.
# ```python
# from rockset import Regions
# RocksetVectorStore(api_server=Regions.euc1a1)
# ```
# 
# * **client**: Rockset client object to use to execute Rockset requests. If not specified, a client object is internally constructed with the `api_key` parameter (or `ROCKSET_API_SERVER` environment variable) and the `api_server` parameter (or `ROCKSET_API_SERVER` environment variable).
# ```python
# from rockset import RocksetClient
# RocksetVectorStore(client=RocksetClient(api_key="<my key>"))
# ```
# 
# * **embedding_col**: The name of the database field containing embeddings. Defaults to `"embedding"`.
# ```python
# RocksetVectorStore(embedding_col="my_embedding")
# ```
# 
# * **metadata_col**: The name of the database field containing node data. Defaults to `"metadata"`.
# ```python
# RocksetVectorStore(metadata_col="node")
# ```
# 
# * **distance_func**: The metric to measure vector relationship. Defaults to cosine similarity.
# ```python
# RocksetVectorStore(distance_func=RocksetVectorStore.DistanceFunc.DOT_PRODUCT)
# ```
