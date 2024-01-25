#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/Timescalevector.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Timescale Vector Store (PostgreSQL)
# 
# This notebook shows how to use the Postgres vector store `TimescaleVector` to store and query vector embeddings.
# 
# ## What is Timescale Vector?
# **[Timescale Vector](https://www.timescale.com/ai) is PostgreSQL++ for AI applications.**
# 
# Timescale Vector enables you to efficiently store and query millions of vector embeddings in `PostgreSQL`.
# - Enhances `pgvector` with faster and more accurate similarity search on millions of vectors via DiskANN inspired indexing algorithm.
# - Enables fast time-based vector search via automatic time-based partitioning and indexing.
# - Provides a familiar SQL interface for querying vector embeddings and relational data.
# 
# Timescale Vector scales with you from POC to production:
# - Simplifies operations by enabling you to store relational metadata, vector embeddings, and time-series data in a single database.
# - Benefits from rock-solid PostgreSQL foundation with enterprise-grade feature liked streaming backups and replication, high-availability and row-level security.
# - Enables a worry-free experience with enterprise-grade security and compliance.
# 
# ## How to use Timescale Vector
# Timescale Vector is available on [Timescale](https://www.timescale.com/ai), the cloud PostgreSQL platform. (There is no self-hosted version at this time.)
# 
# **LlamaIndex users get a 90-day free trial for Timescale Vector.**
# - To get started, [signup](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch&utm_source=llamaindex&utm_medium=referral) to Timescale, create a new database and follow this notebook!
# - See the [Timescale Vector explainer blog](https://www.timescale.com/blog/how-we-made-postgresql-the-best-vector-database/?utm_campaign=vectorlaunch&utm_source=llamaindex&utm_medium=referral) for details and performance benchmarks.
# - See the [installation instructions](https://github.com/timescale/python-vector) for more details on using Timescale Vector in python.

# ## 0. Setup
# Let's import everything we'll need for this notebook.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# import logging
# import sys

# Uncomment to see debug logs
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import timescale_vector
from llama_index import SimpleDirectoryReader, StorageContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import TimescaleVectorStore
from llama_index.vector_stores.types import VectorStoreQuery, MetadataFilters
import textwrap
import openai

# ### Setup OpenAI API Key
# To create embeddings for documents loaded into the index, let's configure your OpenAI API key:

# Get openAI api key by reading local .env file
# The .env file should contain a line starting with `OPENAI_API_KEY=sk-`
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# OR set it explicitly
# import os
# os.environ["OPENAI_API_KEY"] = "<your key>"
openai.api_key = os.environ["OPENAI_API_KEY"]

# ### Create a PostgreSQL database and get a Timescale service URL
# You need a service url to connect to your Timescale database instance.
# 
# First, launch a new cloud database in [Timescale](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch&utm_source=llamaindex&utm_medium=referral) (sign up for free using the link above).
# 
# To connect to your cloud PostgreSQL database, you'll need your service URI, which can be found in the cheatsheet or `.env` file you downloaded after creating a new database. 
# 
# The URI will look something like this: `postgres://tsdbadmin:<password>@<id>.tsdb.cloud.timescale.com:<port>/tsdb?sslmode=require`

# Get the service url by reading local .env file
# The .env file should contain a line starting with `TIMESCALE_SERVICE_URL=postgresql://`
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

TIMESCALE_SERVICE_URL = os.environ["TIMESCALE_SERVICE_URL"]

# OR set it explicitly
# TIMESCALE_SERVICE_URL = "postgres://tsdbadmin:<password>@<id>.tsdb.cloud.timescale.com:<port>/tsdb?sslmode=require"

# ## 1. Simple Similarity Search with Timescale Vector

# ### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Loading documents
# For this example, we'll use a [SimpleDirectoryReader](https://gpt-index.readthedocs.io/en/stable/examples/data_connectors/simple_directory_reader.html) to load the documents stored in the the `paul_graham_essay` directory. 
# 
# The `SimpleDirectoryReader` is one of LlamaIndex's most commonly used data connectors to read one or multiple files from a directory.

# load sample data from the data directory using a SimpleDirectoryReader
documents = SimpleDirectoryReader("./data/paul_graham").load_data()
print("Document ID:", documents[0].doc_id)

# ### Create a VectorStore Index with the TimescaleVectorStore
# Next, to perform a similarity search, we first create a `TimescaleVector` [vector store](https://gpt-index.readthedocs.io/en/stable/core_modules/data_modules/storage/vector_stores.html) to store our vector embeddings from the essay content. TimescaleVectorStore takes a few arguments, namely the `service_url` which we loaded above, along with a `table_name` which we will be the name of the table that the vectors are stored in.
# 
# Then we create a [Vector Store Index](https://gpt-index.readthedocs.io/en/stable/community/integrations/vector_stores.html#vector-store-index) on the documents backed by Timescale using the previously documents.

# Create a TimescaleVectorStore to store the documents
vector_store = TimescaleVectorStore.from_params(
    service_url=TIMESCALE_SERVICE_URL,
    table_name="paul_graham_essay",
)

# Create a new VectorStoreIndex using the TimescaleVectorStore
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# ### Query the index
# Now that we've indexed the documents in our VectorStore, we can ask questions about our documents in the index by using the default `query_engine`.
# 
# Note you can also configure the query engine to configure the top_k most similar results returned, as well as metadata filters to filter the results by. See the [configure standard query setting section](https://gpt-index.readthedocs.io/en/stable/core_modules/data_modules/index/vector_store_guide.html) for more details.

query_engine = index.as_query_engine()
response = query_engine.query("Did the author work at YC?")

print(textwrap.fill(str(response), 100))

response = query_engine.query("What did the author work on before college?")

print(textwrap.fill(str(response), 100))

# ### Querying existing index

vector_store = TimescaleVectorStore.from_params(
    service_url=TIMESCALE_SERVICE_URL,
    table_name="paul_graham_essay",
)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do before YC?")

print(textwrap.fill(str(response), 100))

# ## 2. Using ANN search indexes to speed up queries
# 
# (Note: These indexes are ANN indexes, and differ from the index concept in LlamaIndex)
# 
# You can speed up similarity queries by creating an index on the embedding column. You should only do this once you have ingested a large part of your data.
# 
# Timescale Vector supports the following indexes:
# - timescale_vector_index: a disk-ann inspired graph index for fast similarity search (default).
# - pgvector's HNSW index: a hierarchical navigable small world graph index for fast similarity search.
# - pgvector's IVFFLAT index: an inverted file index for fast similarity search.
# 
# Important note: In PostgreSQL, each table can only have one index on a particular column. So if you'd like to test the performance of different index types, you can do so either by (1) creating multiple tables with different indexes, (2) creating multiple vector columns in the same table and creating different indexes on each column, or (3) by dropping and recreating the index on the same column and comparing results.

vector_store = TimescaleVectorStore.from_params(
    service_url=TIMESCALE_SERVICE_URL,
    table_name="paul_graham_essay",
)

# Using the `create_index()` function without additional arguments will create a `timescale_vector (DiskANN)` index by default, using the default parameters.

# Create a timescale vector index (DiskANN)
vector_store.create_index()

# You can also specify the parameters for the index. See the Timescale Vector documentation for a full discussion of the different parameters and their effects on performance.

# drop old index
vector_store.drop_index()

# create new timescale vector index (DiskANN) with specified parameters
vector_store.create_index("tsv", max_alpha=1.0, num_neighbors=50)

# Timescale Vector also supports HNSW and ivfflat indexes:

vector_store.drop_index()

# Create an HNSW index
# Note: You don't need to specify m and ef_construction parameters as we set smart defaults.
vector_store.create_index("hnsw", m=16, ef_construction=64)

# Create an IVFFLAT index
# Note: You don't need to specify num_lists and num_records parameters as we set smart defaults.
vector_store.drop_index()
vector_store.create_index("ivfflat", num_lists=20, num_records=1000)

# We recommend using `timescale-vector` or `HNSW` indexes in general.

# drop the ivfflat index
vector_store.drop_index()
# Create a timescale vector index (DiskANN)
vector_store.create_index()

# ## 3. Similarity Search with time-based filtering
# 
# A key use case for Timescale Vector is efficient time-based vector search. Timescale Vector enables this by automatically partitioning vectors (and associated metadata) by time. This allows you to efficiently query vectors by both similarity to a query vector and time.
# 
# Time-based vector search functionality is helpful for applications like:
# - Storing and retrieving LLM response history (e.g. chatbots)
# - Finding the most recent embeddings that are similar to a query vector (e.g recent news).
# - Constraining similarity search to a relevant time range (e.g asking time-based questions about a knowledge base)
# 
# To illustrate how to use TimescaleVector's time-based vector search functionality, we'll use the git log history for TimescaleDB as a sample dataset and ask questions about it. Each git commit entry has a timestamp associated with it, as well as natural language message and other metadata (e.g author, commit hash etc). 
# 
# We'll illustrate how to create nodes with a time-based uuid and how run similarity searches with time range filters using the TimescaleVector vectorstore.

# ### Extract content and metadata from git log CSV file
# 
# First lets load in the git log csv file into a new collection in our PostgreSQL database named `timescale_commits`.
# 
# Note: Since this is a demo, we will only work with the first 1000 records. In practice, you can load as many records as you want.

import pandas as pd
from pathlib import Path

file_path = Path("../data/csv/commit_history.csv")
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Light data cleaning on CSV
df.dropna(inplace=True)
df = df.astype(str)
df = df[:1000]

# Take a look at the data in the csv (optional)
df.head()

# We'll define a helper funciton to create a uuid for a node and associated vector embedding based on its timestamp. We'll use this function to create a uuid for each git log entry.
# 
# Important note: If you are working with documents/nodes and want the current date and time associated with vector for time-based search, you can skip this step. A uuid will be automatically generated when the nodes are added to the table in Timescale Vector by default. In our case, because we want the uuid to be based on the timestamp in the past, we need to create the uuids manually.

from timescale_vector import client

# Function to take in a date string in the past and return a uuid v1
def create_uuid(date_string: str):
    if date_string is None:
        return None
    time_format = "%a %b %d %H:%M:%S %Y %z"
    datetime_obj = datetime.strptime(date_string, time_format)
    uuid = client.uuid_from_time(datetime_obj)
    return str(uuid)

# Helper functions
from typing import List, Tuple

# Helper function to split name and email given an author string consisting of Name Lastname <email>
def split_name(input_string: str) -> Tuple[str, str]:
    if input_string is None:
        return None, None
    start = input_string.find("<")
    end = input_string.find(">")
    name = input_string[:start].strip()
    return name

from datetime import datetime, timedelta

def create_date(input_string: str) -> datetime:
    if input_string is None:
        return None
    # Define a dictionary to map month abbreviations to their numerical equivalents
    month_dict = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }

    # Split the input string into its components
    components = input_string.split()
    # Extract relevant information
    day = components[2]
    month = month_dict[components[1]]
    year = components[4]
    time = components[3]
    timezone_offset_minutes = int(
        components[5]
    )  # Convert the offset to minutes
    timezone_hours = timezone_offset_minutes // 60  # Calculate the hours
    timezone_minutes = (
        timezone_offset_minutes % 60
    )  # Calculate the remaining minutes
    # Create a formatted string for the timestamptz in PostgreSQL format
    timestamp_tz_str = (
        f"{year}-{month}-{day} {time}+{timezone_hours:02}{timezone_minutes:02}"
    )
    return timestamp_tz_str

# Next, we'll define a function to create a `TextNode` for each git log entry. We'll use the helper function `create_uuid()` we defined above to create a uuid for each node based on its timestampe. And we'll use the helper functions `create_date()` and `split_name()` above to extract relevant metadata from the git log entry and add them to the node.

from llama_index.schema import TextNode, NodeRelationship, RelatedNodeInfo

# Create a Node object from a single row of data
def create_node(row):
    record = row.to_dict()
    record_name = split_name(record["author"])
    record_content = (
        str(record["date"])
        + " "
        + record_name
        + " "
        + str(record["change summary"])
        + " "
        + str(record["change details"])
    )
    # Can change to TextNode as needed
    node = TextNode(
        id_=create_uuid(record["date"]),
        text=record_content,
        metadata={
            "commit": record["commit"],
            "author": record_name,
            "date": create_date(record["date"]),
        },
    )
    return node

nodes = [create_node(row) for _, row in df.iterrows()]

# Next we'll create vector embeddings of the content of each node so that we can perform similarity search on the text associated with each node. We'll use the `OpenAIEmbedding` model to create the embeddings.

# Create embeddings for nodes
from llama_index.embeddings import OpenAIEmbedding

embedding_model = OpenAIEmbedding()

for node in nodes:
    node_embedding = embedding_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

# Let's examine the first node in our collection to see what it looks like.

print(nodes[0].get_content(metadata_mode="all"))

print(nodes[0].get_embedding())

# ### Load documents and metadata into TimescaleVector vectorstore
# Now that we have prepared our nodes and added embeddings to them, let's add them into our TimescaleVector vectorstore.
# 
# We'll create a Timescale Vector instance from the list of nodes we created.
# 
# First, we'll define a collection name, which will be the name of our table in the PostgreSQL database. 
# 
# We'll also define a time delta, which we pass to the `time_partition_interval` argument, which will be used to as the interval for partitioning the data by time. Each partition will consist of data for the specified length of time. We'll use 7 days for simplicity, but you can pick whatever value make sense for your use case -- for example if you query recent vectors frequently you might want to use a smaller time delta like 1 day, or if you query vectors over a decade long time period then you might want to use a larger time delta like 6 months or 1 year.
# 
# Then we'll add the nodes to the Timescale Vector vectorstore.

# Create a timescale vector store and add the newly created nodes to it
ts_vector_store = TimescaleVectorStore.from_params(
    service_url=TIMESCALE_SERVICE_URL,
    table_name="li_commit_history",
    time_partition_interval=timedelta(days=7),
)
_ = ts_vector_store.add(nodes)

# ### Querying vectors by time and similarity
# 
# Now that we have loaded our documents into TimescaleVector, we can query them by time and similarity.
# 
# TimescaleVector provides multiple methods for querying vectors by doing similarity search with time-based filtering Let's take a look at each method below.
# 
# First we define a query string and get the vector embedding for the query string.

# Define query and generate embedding for it
query_str = "What's new with TimescaleDB functions?"
embed_model = OpenAIEmbedding()
query_embedding = embed_model.get_query_embedding(query_str)

# Then we set some variables which we'll use in our time filters.

# Time filter variables for query
start_dt = datetime(
    2023, 8, 1, 22, 10, 35
)  # Start date = 1 August 2023, 22:10:35
end_dt = datetime(
    2023, 8, 30, 22, 10, 35
)  # End date = 30 August 2023, 22:10:35
td = timedelta(days=7)  # Time delta = 7 days

# Method 1: Filter within a provided start date and end date.

# Query the vector database
vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=5
)

# return most similar vectors to query between start date and end date date range
# returns a VectorStoreQueryResult object
query_result = ts_vector_store.query(
    vector_store_query, start_date=start_dt, end_date=end_dt
)
query_result

# Let's inspect the nodes that were returned from the similarity search:

# for each node in the query result, print the node metadata date
for node in query_result.nodes:
    print("-" * 80)
    print(node.metadata["date"])
    print(node.get_content(metadata_mode="all"))

# Note how the query only returns results within the specified date range.

# Method 2: Filter within a provided start date, and a time delta later.
# 

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=5
)

# return most similar vectors to query from start date and a time delta later
query_result = ts_vector_store.query(
    vector_store_query, start_date=start_dt, time_delta=td
)

for node in query_result.nodes:
    print("-" * 80)
    print(node.metadata["date"])
    print(node.get_content(metadata_mode="all"))

# Once again, notice how only nodes between the start date (1 August) and the defined time delta later (7 days later) are returned.

# Method 3: Filter within a provided end date and a time delta earlier.

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=5
)

# return most similar vectors to query from end date and a time delta earlier
query_result = ts_vector_store.query(
    vector_store_query, end_date=end_dt, time_delta=td
)

for node in query_result.nodes:
    print("-" * 80)
    print(node.metadata["date"])
    print(node.get_content(metadata_mode="all"))

# The main takeaway is that in each result above, only vectors within the specified time range are returned. These queries are very efficient as they only need to search the relevant partitions.

# ## 4. Using TimescaleVector store as a Retriever and Query engine 
# 
# Now that we've explored basic similarity search and similarity search with time-based filters, let's look at how to these features of Timescale Vector with LLamaIndex's retriever and query engine.
# 
# First we'll look at how to use TimescaleVector as a [retriever](https://gpt-index.readthedocs.io/en/latest/api_reference/query/retrievers.html), specifically a [Vector Store Retriever](https://gpt-index.readthedocs.io/en/latest/api_reference/query/retrievers/vector_store.html).
# 
# To constrain the nodes retrieved to a relevant time-range, we can use TimescaleVector's time filters. We simply pass the time filter parameters as `vector_strored_kwargs` when creating the retriever.

from llama_index import VectorStoreIndex
from llama_index.storage import StorageContext

index = VectorStoreIndex.from_vector_store(ts_vector_store)
retriever = index.as_retriever(
    vector_store_kwargs=({"start_date": start_dt, "time_delta": td})
)
retriever.retrieve("What's new with TimescaleDB functions?")

# Next we'll look at how to use TimescaleVector as a [query engine](https://gpt-index.readthedocs.io/en/latest/api_reference/query/query_engines.html).
# 
# Once again, we use TimescaleVector's time filters to constrain the search to a relevant time range by passing our time filter parameters as `vector_strored_kwargs` when creating the query engine.

index = VectorStoreIndex.from_vector_store(ts_vector_store)
query_engine = index.as_query_engine(
    vector_store_kwargs=({"start_date": start_dt, "end_date": end_dt})
)

# query_str = "What's new with TimescaleDB? List 3 new features"
query_str = (
    "What's new with TimescaleDB functions? When were these changes made and"
    " by whom?"
)
response = query_engine.query(query_str)
print(str(response))

