#!/usr/bin/env python
# coding: utf-8

# # Jina 8K Context Window Embeddings
# 
# Here we show you how to use `jina-embeddings-v2` which support an 8k context length and is on-par with `text-embedding-ada-002`

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/embeddings/jina_embeddings.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

import nest_asyncio

nest_asyncio.apply()

# ## Setup Embedding Model

from llama_index.embeddings import (
    HuggingFaceEmbedding,
    HuggingFaceInferenceAPIEmbedding,
    OpenAIEmbedding,
)
from llama_index import ServiceContext

# base model
# model_name = "jinaai/jina-embeddings-v2-base-en"
# small model
model_name = "jinaai/jina-embeddings-v2-small-en"

# download model locally
# note: you need enough RAM+compute to run this
embed_model = HuggingFaceEmbedding(
    model_name=model_name, trust_remote_code=True
)

# use inference API on Hugging Face (though you might run into rate limit issues)
# embed_model = HuggingFaceInferenceAPIEmbedding(
#     model_name="jinaai/jina-embeddings-v2-base-en",
# )

# we set chunk size to 1024 for now, you can obviuosly set it to much bigger
service_context = ServiceContext.from_defaults(
    embed_model=embed_model, chunk_size=1024
)

# ### Setup OpenAI ada embeddings as comparison

embed_model_base = OpenAIEmbedding()

service_context_base = ServiceContext.from_defaults(
    embed_model=embed_model_base, chunk_size=1024
)

# ## Setup Index to test this out
# 
# We'll use our standard Paul Graham example.

from llama_index import VectorStoreIndex, SimpleDirectoryReader

reader = SimpleDirectoryReader("../data/paul_graham")
docs = reader.load_data()

index_jina = VectorStoreIndex.from_documents(
    docs, service_context=service_context
)

index_base = VectorStoreIndex.from_documents(
    docs, service_context=service_context_base
)

# ## View Results
# 
# Look at retrieved results with Jina-8k vs. Replicate

from llama_index.response.notebook_utils import #display_source_node

retriever_jina = index_jina.as_retriever(similarity_top_k=1)
retriever_base = index_base.as_retriever(similarity_top_k=1)

retrieved_nodes = retriever_jina.retrieve(
    "What did the author do in art school?"
)

for n in retrieved_nodes:
    #display_source_node(n, source_length=2000)

retrieved_nodes = retriever_base.retrieve("What did the author do in school?")

for n in retrieved_nodes:
    #display_source_node(n, source_length=2000)

