#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/BagelIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # BagelDB
# 
# >[Bagel](https://docs.bageldb.ai/) is a Open Vector Database for AI. It is built for distributed Machine Learning compute. Cutting AI data infra spend by tenfold.
# 
# <a href="https://discord.gg/bA7B6r97" target="_blank">
#       <img src="https://img.shields.io/discord/1073293645303795742" alt="Discord">
#   </a>&nbsp;&nbsp;
# 
# 
# - [Website](https://www.bageldb.ai/)
# - [Documentation](https://docs.bageldb.ai/)
# - [Twitter](https://twitter.com/bageldb_ai)
# - [Discord](https://discord.gg/bA7B6r97)
# 
# 

# 
# ```sh
# pip install betabageldb
# ```
# 
# 
# Like any other database, you can:
# - `.add` 
# - `.get` 
# - `.delete`
# - `.update`
# - `.upsert`
# - `.peek`
# - `.modify`
# - and `.find` runs the similarity search. 

# ## Basic Example
# 

# !pip install llama-index --quiet
# !pip install betabageldb
# !pip install sentence-transformers
# !pip install pydantic==1.10.11

# import
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import BagelVectorStore
from llama_index.storage.storage_context import StorageContext
from IPython.#display import Markdown, #display
import bagel
from bagel import Settings

# set up OpenAI
import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# create server settings
server_settings = Settings(
    bagel_api_impl="rest", bagel_server_host="api.bageldb.ai"
)

# create client
client = bagel.Client(server_settings)

# create collection
collection = client.get_or_create_cluster("testing_embeddings")

# define embedding function
embed_model = "local:BAAI/bge-small-en-v1.5"

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# set up BagelVectorStore and load in data
vector_store = BagelVectorStore(collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

service_context = ServiceContext.from_defaults(embed_model=embed_model)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(f"<b>{response}</b>")

# ## Create - Add - Get

def create_add_get(client):
    """
    Create, add, and get
    """
    name = "testing"

    # Get or create a cluster
    cluster = client.get_or_create_cluster(name)

    # Add documents to the cluster
    resp = cluster.add(
        documents=[
            "This is document1",
            "This is bidhan",
        ],
        metadatas=[{"source": "google"}, {"source": "notion"}],
        ids=[str(uuid.uuid4()), str(uuid.uuid4())],
    )

    # Print count
    print("count of docs:", cluster.count())

    # Get the first item
    first_item = cluster.peek(1)
    if first_item:
        print("get 1st item")

    print(">> create_add_get done !\n")

# ## Create - Add - Find by Text

def create_add_find(client):
    """
    Create, add, & find

    Parameters
    ----------
    api : _type_
        _description_
    """
    name = "testing"

    # Get or create a cluster
    cluster = client.get_or_create_cluster(name)

    # Add documents to the cluster
    cluster.add(
        documents=[
            "This is document",
            "This is Towhid",
            "This is text",
        ],
        metadatas=[
            {"source": "notion"},
            {"source": "notion"},
            {"source": "google-doc"},
        ],
        ids=[str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())],
    )

    # Query the cluster for similar results
    results = cluster.find(
        query_texts=["This"],
        n_results=5,
        where={"source": "notion"},
        where_document={"$contains": "is"},
    )

    print(results)
    print(">> create_add_find done  !\n")

# ## Create - Add - Find by Embeddings

def create_add_find_em(client):
    """Create, add, & find embeddings

    Parameters
    ----------
    api : _type_
        _description_
    """
    name = "testing_embeddings"
    # Reset the Bagel server
    client.reset()

    # Get or create a cluster
    cluster = api.get_or_create_cluster(name)
    # Add embeddings and other data to the cluster
    cluster.add(
        embeddings=[
            [1.1, 2.3, 3.2],
            [4.5, 6.9, 4.4],
            [1.1, 2.3, 3.2],
            [4.5, 6.9, 4.4],
            [1.1, 2.3, 3.2],
            [4.5, 6.9, 4.4],
            [1.1, 2.3, 3.2],
            [4.5, 6.9, 4.4],
        ],
        metadatas=[
            {"uri": "img1.png", "style": "style1"},
            {"uri": "img2.png", "style": "style2"},
            {"uri": "img3.png", "style": "style1"},
            {"uri": "img4.png", "style": "style1"},
            {"uri": "img5.png", "style": "style1"},
            {"uri": "img6.png", "style": "style1"},
            {"uri": "img7.png", "style": "style1"},
            {"uri": "img8.png", "style": "style1"},
        ],
        documents=[
            "doc1",
            "doc2",
            "doc3",
            "doc4",
            "doc5",
            "doc6",
            "doc7",
            "doc8",
        ],
        ids=["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"],
    )

    # Query the cluster for results
    results = cluster.find(query_embeddings=[[1.1, 2.3, 3.2]], n_results=5)

    print("find result:", results)
    print(">> create_add_find_em done  !\n")

# ## Create - Add - Modify - Update

def create_add_modify_update(client):
    """
    Create, add, modify, and update

    Parameters
    ----------
    api : _type_
        _description_
    """
    name = "testing"
    new_name = "new_" + name

    # Get or create a cluster
    cluster = client.get_or_create_cluster(name)

    # Modify the cluster name
    print("Before:", cluster.name)
    cluster.modify(name=new_name)
    print("After:", cluster.name)

    # Add documents to the cluster
    cluster.add(
        documents=[
            "This is document1",
            "This is bidhan",
        ],
        metadatas=[{"source": "notion"}, {"source": "google"}],
        ids=["id1", "id2"],
    )

    # Retrieve document metadata before updating
    print("Before update:")
    print(cluster.get(ids=["id1"]))

    # Update document metadata
    cluster.update(ids=["id1"], metadatas=[{"source": "google"}])

    # Retrieve document metadata after updating
    print("After update source:")
    print(cluster.get(ids=["id1"]))

    print(">> create_add_modify_update done !\n")

# ## Create - Upsert

def create_upsert(client):
    """
    Create and upsert

    Parameters
    ----------
    api : _type_
        _description_
    """
    # Reset the Bagel server
    api.reset()

    name = "testing"

    # Get or create a cluster
    cluster = client.get_or_create_cluster(name)

    # Add documents to the cluster
    cluster.add(
        documents=[
            "This is document1",
            "This is bidhan",
        ],
        metadatas=[{"source": "notion"}, {"source": "google"}],
        ids=["id1", "id2"],
    )

    # Upsert documents in the cluster
    cluster.upsert(
        documents=[
            "This is document",
            "This is google",
        ],
        metadatas=[{"source": "notion"}, {"source": "google"}],
        ids=["id1", "id3"],
    )

    # Print the count of documents in the cluster
    print("Count of documents:", cluster.count())
    print(">> create_upsert done !\n")

