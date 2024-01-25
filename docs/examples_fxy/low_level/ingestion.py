#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/low_level/ingestion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Building Data Ingestion from Scratch
# 

# 
# We use Pinecone as the vector database.
# 
# We will show how to do the following:
# 1. How to load in documents.
# 2. How to use a text splitter to split documents.
# 3. How to **manually** construct nodes from each text chunk.
# 4. [Optional] Add metadata to each Node.
# 5. How to generate embeddings for each text chunk.
# 6. How to insert into a vector database.

# # Pinecone
# 
# You will need a [pinecone.io](https://www.pinecone.io/) api key for this tutorial. You can [sign up for free](https://app.pinecone.io/?sessionType=signup) to get a Starter account.
# 
# If you create a Starter account, you can name your application anything you like.
# 
# Once you have an account, navigate to 'API Keys' in the Pinecone console. You can use the default key or create a new one for this tutorial.
# 
# Save your api key and its environment (`gcp_starter` for free accounts). You will need them below.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# # OpenAI
# 
# You will need an [OpenAI](https://openai.com/) api key for this tutorial. Login to your [platform.openai.com](https://platform.openai.com/) account, click on your profile picture in the upper right corner, and choose 'API Keys' from the menu. Create an API key for this tutorial and save it. You will need it below.

# ## Environment
# 
# First we add our dependencies.

#('pip -q install python-dotenv pinecone-client llama-index pymupdf')

# #### Set Environment Variables
# 
# We create a file for our environment variables. Do not commit this file or share it!
# 
# Note: Google Colabs will let you create but not open a .env

dotenv_path = (
    "env"  # Google Colabs will not let you open a .env, but you can set
)
with open(dotenv_path, "w") as f:
    f.write('PINECONE_API_KEY="<your api key>"\n')
    f.write('PINECONE_ENVIRONMENT="gcp-starter"\n')
    f.write('OPENAI_API_KEY="<your api key>"\n')

# Set your OpenAI api key, and Pinecone api key and environment in the file we created.

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=dotenv_path)

# ## Setup
# 
# We build an empty Pinecone Index, and define the necessary LlamaIndex wrappers/abstractions so that we can start loading data into Pinecone.
# 
# 
# Note: Do not save your API keys in the code or add pinecone_env to your repo!

import pinecone

api_key = os.environ["PINECONE_API_KEY"]
environment = os.environ["PINECONE_ENVIRONMENT"]
pinecone.init(api_key=api_key, environment=environment)

index_name = "llamaindex-rag-fs"

# [Optional] Delete the index before re-running the tutorial.
# pinecone.delete_index(index_name)

# dimensions are for text-embedding-ada-002
pinecone.create_index(
    index_name, dimension=1536, metric="euclidean", pod_type="p1"
)

pinecone_index = pinecone.Index(index_name)

# [Optional] drop contents in index - will not work on free accounts
pinecone_index.delete(deleteAll=True)

# #### Create PineconeVectorStore
# 
# Simple wrapper abstraction to use in LlamaIndex. Wrap in StorageContext so we can easily load in Nodes.

from llama_index.vector_stores import PineconeVectorStore

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# ## Build an Ingestion Pipeline from Scratch
# 
# We show how to build an ingestion pipeline as mentioned in the introduction.
# 
# Note that steps (2) and (3) can be handled via our `NodeParser` abstractions, which handle splitting and node creation.
# 
# For the purposes of this tutorial, we show you how to create these objects manually.

# ### 1. Load Data

#('mkdir data')
#('wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"')

import fitz

file_path = "./data/llama2.pdf"
doc = fitz.open(file_path)

# ### 2. Use a Text Splitter to Split Documents
# 
# Here we import our `SentenceSplitter` to split document texts into smaller chunks, while preserving paragraphs/sentences as much as possible.

from llama_index.node_parser import SentenceSplitter

text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)

text_chunks = []
# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, page in enumerate(doc):
    page_text = page.get_text("text")
    cur_text_chunks = text_splitter.split_text(page_text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

# ### 3. Manually Construct Nodes from Text Chunks
# 
# We convert each chunk into a `TextNode` object, a low-level data abstraction in LlamaIndex that stores content but also allows defining metadata + relationships with other Nodes.
# 
# We inject metadata from the document into each node.
# 
# This essentially replicates logic in our `SentenceSplitter`.

from llama_index.schema import TextNode

nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc_idx = doc_idxs[idx]
    src_page = doc[src_doc_idx]
    nodes.append(node)

print(nodes[0].metadata)

# print a sample node
print(nodes[0].get_content(metadata_mode="all"))

# ### [Optional] 4. Extract Metadata from each Node
# 
# We extract metadata from each Node using our Metadata extractors.
# 
# This will add more metadata to each Node.

from llama_index.extractors import (
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.ingestion import IngestionPipeline
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

extractors = [
    TitleExtractor(nodes=5, llm=llm),
    QuestionsAnsweredExtractor(questions=3, llm=llm),
]

pipeline = IngestionPipeline(
    transformations=extractors,
)
nodes = pipeline.run(nodes=nodes, in_place=False)

print(nodes[0].metadata)

# ### 5. Generate Embeddings for each Node
# 
# Generate document embeddings for each Node using our OpenAI embedding model (`text-embedding-ada-002`).
# 
# Store these on the `embedding` property on each Node.

from llama_index.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding()

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

# ### 6. Load Nodes into a Vector Store
# 
# We now insert these nodes into our `PineconeVectorStore`.
# 
# **NOTE**: We skip the VectorStoreIndex abstraction, which is a higher-level abstraction that handles ingestion as well. We use `VectorStoreIndex` in the next section to fast-track retrieval/querying.

vector_store.add(nodes)

# ## Retrieve and Query from the Vector Store
# 
# Now that our ingestion is complete, we can retrieve/query this vector store.
# 
# **NOTE**: We can use our high-level `VectorStoreIndex` abstraction here. See the next section to see how to define retrieval at a lower-level!

from llama_index import VectorStoreIndex
from llama_index.storage import StorageContext

index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine()

query_str = "Can you tell me about the key concepts for safety finetuning"

response = query_engine.query(query_str)

print(str(response))

