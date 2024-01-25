#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/existing_data/pinecone_existing_data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Guide: Using Vector Store Index with Existing Pinecone Vector Store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import pinecone

api_key = os.environ["PINECONE_API_KEY"]
pinecone.init(api_key=api_key, environment="eu-west1-gcp")

# ## Prepare Sample "Existing" Pinecone Vector Store

# ### Create index

indexes = pinecone.list_indexes()
print(indexes)

if "quickstart-index" not in indexes:
    # dimensions are for text-embedding-ada-002
    pinecone.create_index(
        "quickstart-index", dimension=1536, metric="euclidean", pod_type="p1"
    )

pinecone_index = pinecone.Index("quickstart-index")

pinecone_index.delete(deleteAll="true")

# ### Define sample data
# We create 4 sample books 

books = [
    {
        "title": "To Kill a Mockingbird",
        "author": "Harper Lee",
        "content": (
            "To Kill a Mockingbird is a novel by Harper Lee published in"
            " 1960..."
        ),
        "year": 1960,
    },
    {
        "title": "1984",
        "author": "George Orwell",
        "content": (
            "1984 is a dystopian novel by George Orwell published in 1949..."
        ),
        "year": 1949,
    },
    {
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "content": (
            "The Great Gatsby is a novel by F. Scott Fitzgerald published in"
            " 1925..."
        ),
        "year": 1925,
    },
    {
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "content": (
            "Pride and Prejudice is a novel by Jane Austen published in"
            " 1813..."
        ),
        "year": 1813,
    },
]

# ### Add data
# We add the sample books to our Weaviate "Book" class (with embedding of content field

import uuid
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding()

entries = []
for book in books:
    vector = embed_model.get_text_embedding(book["content"])
    entries.append(
        {"id": str(uuid.uuid4()), "values": vector, "metadata": book}
    )
pinecone_index.upsert(entries)

# ## Query Against "Existing" Pinecone Vector Store 

from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex
from llama_index.response.pprint_utils import pprint_source_node

# You must properly select a class property as the "text" field.

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, text_key="content"
)

retriever = VectorStoreIndex.from_vector_store(vector_store).as_retriever(
    similarity_top_k=1
)

nodes = retriever.retrieve("What is that book about a bird again?")

# Let's inspect the retrieved node. We can see that the book data is loaded as LlamaIndex `Node` objects, with the "content" field as the main text.

pprint_source_node(nodes[0])

# The remaining fields should be loaded as metadata (in `metadata`)

nodes[0].node.metadata

