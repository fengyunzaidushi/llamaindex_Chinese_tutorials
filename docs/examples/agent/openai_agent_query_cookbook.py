#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/agent/openai_agent_query_cookbook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # OpenAI Agent + Query Engine Experimental Cookbook
# 
# 

# 
# - Auto retrieval 
# - Joint SQL and vector search

# ## AutoRetrieval from a Vector Database
# 
# Our existing "auto-retrieval" capabilities (in `VectorIndexAutoRetriever`) allow an LLM to infer the right query parameters for a vector database - including both the query string and metadata filter.
# 
# Since the OpenAI Function API can infer function parameters, we explore its capabilities in performing auto-retrieval here.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import pinecone
import os

api_key = os.environ["PINECONE_API_KEY"]
pinecone.init(api_key=api_key, environment="us-west4-gcp-free")

import os
import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
import openai

openai.api_key = "sk-<your-key>"

# dimensions are for text-embedding-ada-002
try:
    pinecone.create_index(
        "quickstart-index", dimension=1536, metric="euclidean", pod_type="p1"
    )
except Exception:
    # most likely index already exists
    pass

pinecone_index = pinecone.Index("quickstart-index")

# Optional: delete data in your pinecone index
pinecone_index.delete(deleteAll=True, namespace="test")

from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores import PineconeVectorStore

from llama_index.schema import TextNode

nodes = [
    TextNode(
        text=(
            "Michael Jordan is a retired professional basketball player,"
            " widely regarded as one of the greatest basketball players of all"
            " time."
        ),
        metadata={
            "category": "Sports",
            "country": "United States",
            "gender": "male",
            "born": 1963,
        },
    ),
    TextNode(
        text=(
            "Angelina Jolie is an American actress, filmmaker, and"
            " humanitarian. She has received numerous awards for her acting"
            " and is known for her philanthropic work."
        ),
        metadata={
            "category": "Entertainment",
            "country": "United States",
            "gender": "female",
            "born": 1975,
        },
    ),
    TextNode(
        text=(
            "Elon Musk is a business magnate, industrial designer, and"
            " engineer. He is the founder, CEO, and lead designer of SpaceX,"
            " Tesla, Inc., Neuralink, and The Boring Company."
        ),
        metadata={
            "category": "Business",
            "country": "United States",
            "gender": "male",
            "born": 1971,
        },
    ),
    TextNode(
        text=(
            "Rihanna is a Barbadian singer, actress, and businesswoman. She"
            " has achieved significant success in the music industry and is"
            " known for her versatile musical style."
        ),
        metadata={
            "category": "Music",
            "country": "Barbados",
            "gender": "female",
            "born": 1988,
        },
    ),
    TextNode(
        text=(
            "Cristiano Ronaldo is a Portuguese professional footballer who is"
            " considered one of the greatest football players of all time. He"
            " has won numerous awards and set multiple records during his"
            " career."
        ),
        metadata={
            "category": "Sports",
            "country": "Portugal",
            "gender": "male",
            "born": 1985,
        },
    ),
]

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, namespace="test"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(nodes, storage_context=storage_context)

# #### Define Function Tool
# 
# Here we define the function interface, which is passed to OpenAI to perform auto-retrieval.
# 
# We were not able to get OpenAI to work with nested pydantic objects or tuples as arguments,
# so we converted the metadata filter keys and values into lists for the function API to work with.

# define function tool
from llama_index.tools import FunctionTool
from llama_index.vector_stores.types import (
    VectorStoreInfo,
    MetadataInfo,
    MetadataFilter,
    MetadataFilters,
    FilterCondition,
    FilterOperator,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

from typing import List, Tuple, Any
from pydantic import BaseModel, Field

# hardcode top k for now
top_k = 3

# define vector store info describing schema of vector store
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
        MetadataInfo(
            name="gender",
            type="str",
            description=("Gender of the celebrity, one of [male, female]"),
        ),
        MetadataInfo(
            name="born",
            type="int",
            description=("Born year of the celebrity, could be any integer"),
        ),
    ],
)

# define pydantic model for auto-retrieval function
class AutoRetrieveModel(BaseModel):
    query: str = Field(..., description="natural language query string")
    filter_key_list: List[str] = Field(
        ..., description="List of metadata filter field names"
    )
    filter_value_list: List[Any] = Field(
        ...,
        description=(
            "List of metadata filter field values (corresponding to names"
            " specified in filter_key_list)"
        ),
    )
    filter_operator_list: List[str] = Field(
        ...,
        description=(
            "Metadata filters conditions (could be one of <, <=, >, >=, ==, !=)"
        ),
    )
    filter_condition: str = Field(
        ...,
        description=("Metadata filters condition values (could be AND or OR)"),
    )

description = f"""\
Use this tool to look up biographical information about celebrities.
The vector database schema is given below:
{vector_store_info.json()}
"""

# Define AutoRetrieve Functions

def auto_retrieve_fn(
    query: str,
    filter_key_list: List[str],
    filter_value_list: List[any],
    filter_operator_list: List[str],
    filter_condition: str,
):
    """Auto retrieval function.

    Performs auto-retrieval from a vector database, and then applies a set of filters.

    """
    query = query or "Query"

    metadata_filters = [
        MetadataFilter(key=k, value=v, operator=op)
        for k, v, op in zip(
            filter_key_list, filter_value_list, filter_operator_list
        )
    ]
    retriever = VectorIndexRetriever(
        index,
        filters=MetadataFilters(
            filters=metadata_filters, condition=filter_condition
        ),
        top_k=top_k,
    )
    query_engine = RetrieverQueryEngine.from_args(retriever)

    response = query_engine.query(query)
    return str(response)

auto_retrieve_tool = FunctionTool.from_defaults(
    fn=auto_retrieve_fn,
    name="celebrity_bios",
    description=description,
    fn_schema=AutoRetrieveModel,
)

# ###

from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI

agent = OpenAIAgent.from_tools(
    [auto_retrieve_tool],
    llm=OpenAI(temperature=0, model="gpt-4-0613"),
    verbose=True,
)

response = agent.chat("Tell me about two celebrities from the United States. ")
print(str(response))

response = agent.chat("Tell me about two celebrities born after 1980. ")
print(str(response))

response = agent.chat(
    "Tell me about few celebrities under category business and born after 1950. "
)
print(str(response))

# ## Joint Text-to-SQL and Semantic Search
# 
# This is currently handled by our `SQLAutoVectorQueryEngine`.
# 
# Let's try implementing this by giving our `OpenAIAgent` access to two query tools: SQL and Vector 

# #### Load and Index Structured Data
# 
# We load sample structured datapoints into a SQL db and index it.

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
from llama_index import SQLDatabase, SQLStructStoreIndex

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

sql_database = SQLDatabase(engine, include_tables=["city_stats"])

from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["city_stats"],
)

# #### Load and Index Unstructured Data
# 
# We load unstructured data into a vector index backed by Pinecone

# install wikipedia python package
#('pip install wikipedia')

from llama_index import (
    WikipediaReader,
    SimpleDirectoryReader,
    VectorStoreIndex,
)

cities = ["Toronto", "Berlin", "Tokyo"]
wiki_docs = WikipediaReader().load_data(pages=cities)

# define pinecone index
import pinecone
import os

api_key = os.environ["PINECONE_API_KEY"]
pinecone.init(api_key=api_key, environment="us-west1-gcp")

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
llm = OpenAI(temperature=0, model="gpt-4")
service_context = ServiceContext.from_defaults(chunk_size=chunk_size, llm=llm)
node_parser = TokenTextSplitter(chunk_size=chunk_size)

# define pinecone vector index
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, namespace="wiki_cities"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex([], storage_context=storage_context)

# Each document has metadata of the city attached
for city, wiki_doc in zip(cities, wiki_docs):
    nodes = node_parser.get_nodes_from_documents([wiki_doc])
    # add metadata to each node
    for node in nodes:
        node.metadata = {"title": city}
    vector_index.insert_nodes(nodes)

# #### Define Query Engines / Tools

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
    query_engine=query_engine,
    name="sql_tool",
    description=(
        "Useful for translating a natural language query into a SQL query over"
        " a table containing: city_stats, containing the population/country of"
        " each city"
    ),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=retriever_query_engine,
    name="vector_tool",
    description=(
        f"Useful for answering semantic questions about different cities"
    ),
)

# ###

from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI

agent = OpenAIAgent.from_tools(
    [sql_tool, vector_tool],
    llm=OpenAI(temperature=0, model="gpt-4-0613"),
    verbose=True,
)

# NOTE: gpt-3.5 gives the wrong answer, but gpt-4 is able to reason over both loops
response = agent.chat(
    "Tell me about the arts and culture of the city with the highest"
    " population"
)
print(str(response))

response = agent.chat("Tell me about the history of Berlin")
print(str(response))

response = agent.chat(
    "Can you give me the country corresponding to each city?"
)
print(str(response))

