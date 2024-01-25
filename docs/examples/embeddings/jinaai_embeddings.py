#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/embeddings/jinaai_embeddings.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Jina Embeddings

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# You may also need other packages that do not come direcly with llama-index

#('pip install Pillow')

# For this example, you will need an API key which you can get from https://jina.ai/embeddings/

import os

jinaai_api_key = "YOUR_JINAAI_API_KEY"
os.environ["JINAAI_API_KEY"] = jinaai_api_key

# ## Embed text and queries with Jina embedding models through JinaAI API

# You can encode your text and your queries using the JinaEmbedding class

from llama_index.embeddings.jinaai import JinaEmbedding

embed_model = JinaEmbedding(
    api_key=jinaai_api_key,
    model="jina-embeddings-v2-base-en",
)

embeddings = embed_model.get_text_embedding("This is the text to embed")

print(len(embeddings))
print(embeddings[:5])

embeddings = embed_model.get_query_embedding("This is the query to embed")
print(len(embeddings))
print(embeddings[:5])

# #### Embed in batches

# You can also embed text in batches, the batch size can be controlled by setting the `embed_batch_size` parameter (the default value will be 10 if not passed, and it should not be larger than 2048)

embed_model = JinaEmbedding(
    api_key=jinaai_api_key,
    model="jina-embeddings-v2-base-en",
    embed_batch_size=16,
)

embeddings = embed_model.get_text_embedding_batch(
    ["This is the text to embed", "More text can be provided in a batch"]
)

print(len(embeddings))
print(embeddings[0][:5])

# ## Let's build a RAG pipeline using Jina AI Embeddings

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Imports

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)

from llama_index.llms import OpenAI
from llama_index.response.notebook_utils import #display_source_node

from IPython.#display import Markdown, #display

# #### Load Data

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# #### Build index

your_openai_key = "YOUR_OPENAI_KEY"
llm = OpenAI(api_key=your_openai_key)
embed_model = JinaEmbedding(
    api_key=jinaai_api_key,
    model="jina-embeddings-v2-base-en",
    embed_batch_size=16,
)

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model
)
index = VectorStoreIndex.from_documents(
    documents=documents, service_context=service_context
)

# #### Build retriever

search_query_retriever = index.as_retriever(service_context=service_context)

search_query_retrieved_nodes = search_query_retriever.retrieve(
    "What happened after the thesis?"
)

for n in search_query_retrieved_nodes:
    #display_source_node(n, source_length=2000)

