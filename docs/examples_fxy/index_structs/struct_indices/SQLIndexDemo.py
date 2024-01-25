#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/index_structs/struct_indices/SQLIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Text-to-SQL Guide (Query Engine + Retriever)
# 
# This is a basic guide to LlamaIndex's Text-to-SQL capabilities. 
# 1. We first show how to perform text-to-SQL over a toy dataset: this will do "retrieval" (sql query over db) and "synthesis".
# 2. We then show how to buid a TableIndex over the schema to dynamically retrieve relevant tables during query-time.
# 3. We finally show you how to define a text-to-SQL retriever on its own. 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-.."
openai.api_key = os.environ["OPENAI_API_KEY"]

# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from IPython.#display import Markdown, #display

# ### Create Database Schema
# 
# We use `sqlalchemy`, a popular SQL database toolkit, to create an empty `city_stats` Table

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
)

engine = create_engine("sqlite:///:memory:")
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

# ### Define SQL Database
# 
# We first define our `SQLDatabase` abstraction (a light wrapper around SQLAlchemy). 

from llama_index import SQLDatabase, ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

sql_database = SQLDatabase(engine, include_tables=["city_stats"])

# We add some testing data to our SQL database.

sql_database = SQLDatabase(engine, include_tables=["city_stats"])
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

# view current table
stmt = select(
    city_stats_table.c.city_name,
    city_stats_table.c.population,
    city_stats_table.c.country,
).select_from(city_stats_table)

with engine.connect() as connection:
    results = connection.execute(stmt).fetchall()
    print(results)

# ### Query Index

# We first show how we can execute a raw SQL query, which directly executes over the table.

from sqlalchemy import text

with engine.connect() as con:
    rows = con.execute(text("SELECT city_name from city_stats"))
    for row in rows:
        print(row)

# ## Part 1: Text-to-SQL Query Engine
# Once we have constructed our SQL database, we can use the NLSQLTableQueryEngine to
# construct natural language queries that are synthesized into SQL queries.
# 
# Note that we need to specify the tables we want to use with this query engine.
# If we don't the query engine will pull all the schema context, which could
# overflow the context window of the LLM.

from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["city_stats"],
)
query_str = "Which city has the highest population?"
response = query_engine.query(query_str)

#display(Markdown(f"<b>{response}</b>"))

# This query engine should be used in any case where you can specify the tables you want
# to query over beforehand, or the total size of all the table schema plus the rest of
# the prompt fits your context window.

# ## Part 2: Query-Time Retrieval of Tables for Text-to-SQL
# If we don't know ahead of time which table we would like to use, and the total size of
# the table schema overflows your context window size, we should store the table schema 
# in an index so that during query time we can retrieve the right schema.
# 
# The way we can do this is using the SQLTableNodeMapping object, which takes in a 
# SQLDatabase and produces a Node object for each SQLTableSchema object passed 
# into the ObjectIndex constructor.
# 

from llama_index.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index import VectorStoreIndex

# set Logging to DEBUG for more detailed outputs
table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    (SQLTableSchema(table_name="city_stats"))
]  # add a SQLTableSchema for each table

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=1)
)

# Now we can take our SQLTableRetrieverQueryEngine and query it for our response.

response = query_engine.query("Which city has the highest population?")
#display(Markdown(f"<b>{response}</b>"))

# you can also fetch the raw result from SQLAlchemy!
response.metadata["result"]

# You can also add additional context information for each table schema you define.

# manually set context text
city_stats_text = (
    "This table gives information regarding the population and country of a"
    " given city.\nThe user will query with codewords, where 'foo' corresponds"
    " to population and 'bar'corresponds to city."
)

table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    (SQLTableSchema(table_name="city_stats", context_str=city_stats_text))
]

# ## Part 3: Text-to-SQL Retriever
# 
# So far our text-to-SQL capability is packaged in a query engine and consists of both retrieval and synthesis.
# 
# You can use the SQL retriever on its own. We show you some different parameters you can try, and also show how to plug it into our `RetrieverQueryEngine` to get roughly the same results.

from llama_index.retrievers import NLSQLRetriever

# default retrieval (return_raw=True)
nl_sql_retriever = NLSQLRetriever(
    sql_database, tables=["city_stats"], return_raw=True
)

results = nl_sql_retriever.retrieve(
    "Return the top 5 cities (along with their populations) with the highest population."
)

from llama_index.response.notebook_utils import #display_source_node

for n in results:
    #display_source_node(n)

# default retrieval (return_raw=False)
nl_sql_retriever = NLSQLRetriever(
    sql_database, tables=["city_stats"], return_raw=False
)

results = nl_sql_retriever.retrieve(
    "Return the top 5 cities (along with their populations) with the highest population."
)

# NOTE: all the content is in the metadata
for n in results:
    #display_source_node(n, show_source_metadata=True)

# ### Plug into our `RetrieverQueryEngine`
# 
# We compose our SQL Retriever with our standard `RetrieverQueryEngine` to synthesize a response. The result is roughly similar to our packaged `Text-to-SQL` query engines.

from llama_index.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(nl_sql_retriever)

response = query_engine.query(
    "Return the top 5 cities (along with their populations) with the highest population."
)

print(str(response))

