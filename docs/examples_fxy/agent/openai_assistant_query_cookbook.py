#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/agent/openai_assistant_query_cookbook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # OpenAI Assistant Advanced Retrieval Cookbook
# 
# 

# 
# - Joint QA + Summarization
# - Auto retrieval 
# - Joint SQL and vector search

#('pip install llama-index')

import nest_asyncio

nest_asyncio.apply()

# ## Joint QA and Summarization
# 

# ### Load Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

from llama_index import SimpleDirectoryReader

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# ### Setup Vector + Summary Indexes/Query Engines/Tools

from llama_index.llms import OpenAI
from llama_index import (
    ServiceContext,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
)

# initialize service context (set chunk size)
llm = OpenAI()
service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm)
nodes = service_context.node_parser.get_nodes_from_documents(documents)

# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# Define Summary Index and Vector Index over Same Data
summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

# define query engines
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()

from llama_index.tools.query_engine import QueryEngineTool

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    name="summary_tool",
    description=(
        "Useful for summarization questions related to the author's life"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    name="vector_tool",
    description=(
        "Useful for retrieving specific context to answer specific questions about the author's life"
    ),
)

# ### Define Assistant Agent

from llama_index.agent import OpenAIAssistantAgent

agent = OpenAIAssistantAgent.from_new(
    name="QA bot",
    instructions="You are a bot designed to answer questions about the author",
    openai_tools=[],
    tools=[summary_tool, vector_tool],
    verbose=True,
    run_retrieve_sleep_time=1.0,
)

# #### Results: A bit flaky

response = agent.chat("Can you give me a summary about the author's life?")
print(str(response))

response = agent.query("What did the author do after RICS?")
print(str(response))

# ## AutoRetrieval from a Vector Database
# 
# Our existing "auto-retrieval" capabilities (in `VectorIndexAutoRetriever`) allow an LLM to infer the right query parameters for a vector database - including both the query string and metadata filter.
# 
# Since the Assistant API can call functions + infer function parameters, we explore its capabilities in performing auto-retrieval here.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

import pinecone
import os

api_key = os.environ["PINECONE_API_KEY"]
pinecone.init(api_key=api_key, environment="us-west1-gcp")

# dimensions are for text-embedding-ada-002
try:
    pinecone.create_index(
        "quickstart", dimension=1536, metric="euclidean", pod_type="p1"
    )
except Exception:
    # most likely index already exists
    pass

pinecone_index = pinecone.Index("quickstart")

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
    ExactMatchFilter,
    MetadataFilters,
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
    ],
)

# define pydantic model for auto-retrieval function
class AutoRetrieveModel(BaseModel):
    query: str = Field(..., description="natural language query string")
    filter_key_list: List[str] = Field(
        ..., description="List of metadata filter field names"
    )
    filter_value_list: List[str] = Field(
        ...,
        description=(
            "List of metadata filter field values (corresponding to names"
            " specified in filter_key_list)"
        ),
    )

def auto_retrieve_fn(
    query: str, filter_key_list: List[str], filter_value_list: List[str]
):
    """Auto retrieval function.

    Performs auto-retrieval from a vector database, and then applies a set of filters.

    """
    query = query or "Query"

    exact_match_filters = [
        ExactMatchFilter(key=k, value=v)
        for k, v in zip(filter_key_list, filter_value_list)
    ]
    retriever = VectorIndexRetriever(
        index,
        filters=MetadataFilters(filters=exact_match_filters),
        top_k=top_k,
    )
    results = retriever.retrieve(query)
    return [r.get_content() for r in results]

description = f"""\
Use this tool to look up biographical information about celebrities.
The vector database schema is given below:
{vector_store_info.json()}
"""

auto_retrieve_tool = FunctionTool.from_defaults(
    fn=auto_retrieve_fn,
    name="celebrity_bios",
    description=description,
    fn_schema=AutoRetrieveModel,
)

auto_retrieve_fn(
    "celebrity from the United States",
    filter_key_list=["country"],
    filter_value_list=["United States"],
)

# ###

from llama_index.agent import OpenAIAssistantAgent

agent = OpenAIAssistantAgent.from_new(
    name="Celebrity bot",
    instructions="You are a bot designed to answer questions about celebrities.",
    tools=[auto_retrieve_tool],
    verbose=True,
)

response = agent.chat("Tell me about two celebrities from the United States. ")
print(str(response))

# ## Joint Text-to-SQL and Semantic Search
# 
# This is currenty handled by our `SQLAutoVectorQueryEngine`.
# 
# Let's try implementing this by giving our `OpenAIAssistantAgent` access to two query tools: SQL and Vector search.

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

from llama_index.node_parser import SimpleNodeParser
from llama_index import ServiceContext
from llama_index.storage import StorageContext
from llama_index.text_splitter import TokenTextSplitter
from llama_index.llms import OpenAI

# define node parser and LLM
chunk_size = 1024
llm = OpenAI(temperature=0, model="gpt-4")
service_context = ServiceContext.from_defaults(chunk_size=chunk_size, llm=llm)
text_splitter = TokenTextSplitter(chunk_size=chunk_size)
node_parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)

# use default in-memory store
storage_context = StorageContext.from_defaults()
vector_index = VectorStoreIndex([], storage_context=storage_context)

# Each document has metadata of the city attached
for city, wiki_doc in zip(cities, wiki_docs):
    nodes = node_parser.get_nodes_from_documents([wiki_doc])
    # add metadata to each node
    for node in nodes:
        node.metadata = {"title": city}
    vector_index.insert_nodes(nodes)

# #### Define Query Engines / Tools

from llama_index.tools.query_engine import QueryEngineTool

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
    query_engine=vector_index.as_query_engine(similarity_top_k=2),
    name="vector_tool",
    description=(
        f"Useful for answering semantic questions about different cities"
    ),
)

# ###

from llama_index.agent import OpenAIAssistantAgent

agent = OpenAIAssistantAgent.from_new(
    name="City bot",
    instructions="You are a bot designed to answer questions about cities (both unstructured and structured data)",
    tools=[sql_tool, vector_tool],
    verbose=True,
)

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

