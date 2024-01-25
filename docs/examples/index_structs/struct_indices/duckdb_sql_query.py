#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/index_structs/struct_indices/duckdb_sql_query.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # SQL Query Engine with LlamaIndex + DuckDB
# 
# This guide showcases the core LlamaIndex SQL capabilities with DuckDB. 
# 
# We go through some core LlamaIndex data structures, including the `NLSQLTableQueryEngine` and `SQLTableRetrieverQueryEngine`. 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

#('pip install duckdb duckdb-engine')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    SQLDatabase,
    SimpleDirectoryReader,
    WikipediaReader,
    Document,
)
from llama_index.indices.struct_store import (
    NLSQLTableQueryEngine,
    SQLTableRetrieverQueryEngine,
)

from IPython.#display import Markdown, #display

# ## Basic Text-to-SQL with our `NLSQLTableQueryEngine` 
# 

# ### Create Database Schema + Test Data
# 
# We use sqlalchemy, a popular SQL database toolkit, to connect to DuckDB and create an empty `city_stats` Table. We then populate it with some test data.
# 

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

engine = create_engine("duckdb:///:memory:")
# uncomment to make this work with MotherDuck
# engine = create_engine("duckdb:///md:llama-index")
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
    {
        "city_name": "Chicago",
        "population": 2679000,
        "country": "United States",
    },
    {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

with engine.connect() as connection:
    cursor = connection.exec_driver_sql("SELECT * FROM city_stats")
    print(cursor.fetchall())

# ### Create SQLDatabase Object
# 
# We first define our SQLDatabase abstraction (a light wrapper around SQLAlchemy).

from llama_index import SQLDatabase

sql_database = SQLDatabase(engine, include_tables=["city_stats"])

# ### Query Index

# Here we demonstrate the capabilities of `NLSQLTableQueryEngine`, which performs text-to-SQL.
# 
# 1. We construct a `NLSQLTableQueryEngine` and pass in our SQL database object.
# 2. We run queries against the query engine.

query_engine = NLSQLTableQueryEngine(sql_database)

response = query_engine.query("Which city has the highest population?")

str(response)

response.metadata

# ## Advanced Text-to-SQL with our `SQLTableRetrieverQueryEngine` 
# 

# 
# We first index the schemas with our `ObjectIndex`, and then use our `SQLTableRetrieverQueryEngine` abstraction on top.

engine = create_engine("duckdb:///:memory:")
# uncomment to make this work with MotherDuck
# engine = create_engine("duckdb:///md:llama-index")
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
all_table_names = ["city_stats"]
# create a ton of dummy tables
n = 100
for i in range(n):
    tmp_table_name = f"tmp_table_{i}"
    tmp_table = Table(
        tmp_table_name,
        metadata_obj,
        Column(f"tmp_field_{i}_1", String(16), primary_key=True),
        Column(f"tmp_field_{i}_2", Integer),
        Column(f"tmp_field_{i}_3", String(16), nullable=False),
    )
    all_table_names.append(f"tmp_table_{i}")

metadata_obj.create_all(engine)

# insert dummy data
from sqlalchemy import insert

rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {
        "city_name": "Chicago",
        "population": 2679000,
        "country": "United States",
    },
    {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

sql_database = SQLDatabase(engine, include_tables=["city_stats"])

# ### Construct Object Index

from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index import VectorStoreIndex

table_node_mapping = SQLTableNodeMapping(sql_database)

table_schema_objs = []
for table_name in all_table_names:
    table_schema_objs.append(SQLTableSchema(table_name=table_name))

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)

# ### Query Index with `SQLTableRetrieverQueryEngine`

query_engine = SQLTableRetrieverQueryEngine(
    sql_database,
    obj_index.as_retriever(similarity_top_k=1),
)

response = query_engine.query("Which city has the highest population?")

response

