#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/embeddings/gemini.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Google Gemini Embeddings

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#("pip install llama-index 'google-generativeai>=0.3.0' matplotlib")

import os

GOOGLE_API_KEY = ""  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# imports
from llama_index.embeddings import GeminiEmbedding

# get API key and create embeddings

model_name = "models/embedding-001"

embed_model = GeminiEmbedding(
    model_name=model_name, api_key=GOOGLE_API_KEY, title="this is a document"
)

embeddings = embed_model.get_text_embedding("Google Gemini Embeddings.")

print(f"Dimension of embeddings: {len(embeddings)}")

embeddings[:5]

embeddings = embed_model.get_query_embedding("Google Gemini Embeddings.")
embeddings[:5]

embeddings = embed_model.get_text_embedding(
    ["Google Gemini Embeddings.", "Google is awesome."]
)

print(f"Dimension of embeddings: {len(embeddings)}")
print(embeddings[0][:5])
print(embeddings[1][:5])

embedding = await embed_model.aget_text_embedding("Google Gemini Embeddings.")
print(embedding[:5])

embeddings = await embed_model.aget_text_embedding_batch(
    [
        "Google Gemini Embeddings.",
        "Google is awesome.",
        "Llamaindex is awesome.",
    ]
)
print(embeddings[0][:5])
print(embeddings[1][:5])
print(embeddings[2][:5])

embedding = await embed_model.aget_query_embedding("Google Gemini Embeddings.")
print(embedding[:5])

