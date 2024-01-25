#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/query_engine/SQLRouterQueryEngine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # SQL Router Query Engine

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

# build a separate vector index per city
# You could also choose to define a single vector index across all docs, and annotate each chunk by metadata
vector_indices = []
for wiki_doc in wiki_docs:
    vector_index = VectorStoreIndex.from_documents([wiki_doc])
    vector_indices.append(vector_index)

# ### Define Query Engines, Set as Tools

vector_query_engines = [index.as_query_engine() for index in vector_indices]

from llama_index.tools.query_engine import QueryEngineTool

sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    description=(
        "Useful for translating a natural language query into a SQL query over"
        " a table containing: city_stats, containing the population/country of"
        " each city"
    ),
)
vector_tools = []
for city, query_engine in zip(cities, vector_query_engines):
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        description=f"Useful for answering semantic questions about {city}",
    )
    vector_tools.append(vector_tool)

# ### Define Router Query Engine

from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.llm_selectors import LLMSingleSelector

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=([sql_tool] + vector_tools),
)

response = query_engine.query("Which city has the highest population?")
print(str(response))

response = query_engine.query("Tell me about the historical museums in Berlin")
print(str(response))

response = query_engine.query("Which countries are each city from?")
print(str(response))

