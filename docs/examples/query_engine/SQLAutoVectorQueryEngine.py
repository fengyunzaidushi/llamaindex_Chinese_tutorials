#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/query_engine/SQLAutoVectorQueryEngine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # SQL Auto Vector Query Engine

# 
# This query engine allows you to combine insights from your structured tables with your unstructured data.
# It first decides whether to query your structured tables for insights.
# Once it does, it can then infer a corresponding query to the vector store in order to fetch corresponding documents.

import openai
import os

os.environ["OPENAI_API_KEY"] = "[You API key]"
openai.api_key = os.environ["OPENAI_API_KEY"]

# ### Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# NOTE: This is ONLY necessary in jupyter notebook.
# Details: Jupyter runs an event-loop behind the scenes.
#          This results in nested event-loops when we start an event-loop to make async queries.
#          This is normally not allowed, we use nest_asyncio to allow it for convenience.
import nest_asyncio

nest_asyncio.apply()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    SQLDatabase,
    WikipediaReader,
)

# ### Create Common Objects
# 
# This includes a `ServiceContext` object containing abstractions such as the LLM and chunk size.
# This also includes a `StorageContext` object containing our vector store abstractions.

# define pinecone index
import pinecone
import os

api_key = os.environ["PINECONE_API_KEY"]
pinecone.init(api_key=api_key, environment="us-west1-gcp-free")

# dimensions are for text-embedding-ada-002
# pinecone.create_index("quickstart", dimension=1536, metric="euclidean", pod_type="p1")
pinecone_index = pinecone.Index("quickstart")

# OPTIONAL: delete all
pinecone_index.delete(deleteAll=True)

from llama_index import ServiceContext
from llama_index.storage import StorageContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.node_parser import TokenTextSplitter
from llama_index.llms import OpenAI

# define node parser and LLM
chunk_size = 1024
llm = OpenAI(temperature=0, model="gpt-4", streaming=True)
service_context = ServiceContext.from_defaults(chunk_size=chunk_size, llm=llm)
node_parser = TokenTextSplitter(chunk_size=chunk_size)

# define pinecone vector index
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, namespace="wiki_cities"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex([], storage_context=storage_context)

# ### Create Database Schema + Test Data
# 
# Here we introduce a toy scenario where there are 100 tables (too big to fit into the prompt)

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)

engine = create_engine("sqlite:///:memory:", future=True)
metadata_obj = MetaData()

# create city SQL table
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)

metadata_obj.create_all(engine)

# print tables
metadata_obj.tables.keys()

# We introduce some test data into the `city_stats` table

from sqlalchemy import insert

rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {"city_name": "Berlin", "population": 3645000, "country": "Germany"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

with engine.connect() as connection:
    cursor = connection.exec_driver_sql("SELECT * FROM city_stats")
    print(cursor.fetchall())

# ### Load Data
# 
# We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.

# install wikipedia python package
#('pip install wikipedia')

cities = ["Toronto", "Berlin", "Tokyo"]
wiki_docs = WikipediaReader().load_data(pages=cities)

# ### Build SQL Index

sql_database = SQLDatabase(engine, include_tables=["city_stats"])

from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine

sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["city_stats"],
)

# ### Build Vector Index

# Each document has metadata of the city attached
for city, wiki_doc in zip(cities, wiki_docs):
    nodes = node_parser.get_nodes_from_documents([wiki_doc])
    # add metadata to each node
    for node in nodes:
        node.metadata = {"title": city}
    vector_index.insert_nodes(nodes)

# ### Define Query Engines, Set as Tools

from llama_index.query_engine import (
    SQLAutoVectorQueryEngine,
    RetrieverQueryEngine,
)
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.indices.vector_store import VectorIndexAutoRetriever

from llama_index.indices.vector_store.retrievers import (
    VectorIndexAutoRetriever,
)
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)

vector_store_info = VectorStoreInfo(
    content_info="articles about different cities",
    metadata_info=[
        MetadataInfo(
            name="title", type="str", description="The name of the city"
        ),
    ],
)
vector_auto_retriever = VectorIndexAutoRetriever(
    vector_index, vector_store_info=vector_store_info
)

retriever_query_engine = RetrieverQueryEngine.from_args(
    vector_auto_retriever, service_context=service_context
)

sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    description=(
        "Useful for translating a natural language query into a SQL query over"
        " a table containing: city_stats, containing the population/country of"
        " each city"
    ),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=retriever_query_engine,
    description=(
        f"Useful for answering semantic questions about different cities"
    ),
)

# ### Define SQLAutoVectorQueryEngine

query_engine = SQLAutoVectorQueryEngine(
    sql_tool, vector_tool, service_context=service_context
)

response = query_engine.query(
    "Tell me about the arts and culture of the city with the highest"
    " population"
)

print(str(response))

response = query_engine.query("Tell me about the history of Berlin")

print(str(response))

response = query_engine.query(
    "Can you give me the country corresponding to each city?"
)

print(str(response))

