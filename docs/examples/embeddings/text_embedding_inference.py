#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/embeddings/text_embedding_inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Text Embedding Inference
# 
# This notebook demonstrates how to configure `TextEmbeddingInference` embeddings.
# 
# The first step is to deploy the embeddings server. For detailed instructions, see the [official repository for Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference).
# 
# Once deployed, the code below will connect to and submit embeddings for inference.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.embeddings import TextEmbeddingsInference

embed_model = TextEmbeddingsInference(
    model_name="BAAI/bge-large-en-v1.5",  # required for formatting inference text,
    timeout=60,  # timeout in seconds
    embed_batch_size=10,  # batch size for embedding
)

embeddings = embed_model.get_text_embedding("Hello World!")
print(len(embeddings))
print(embeddings[:5])

embeddings = await embed_model.aget_text_embedding("Hello World!")
print(len(embeddings))
print(embeddings[:5])

