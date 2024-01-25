#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/embeddings/bedrock.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Bedrock Embeddings
# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

import os

from llama_index.embeddings import BedrockEmbedding

embed_model = BedrockEmbedding.from_credentials(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    aws_region="<aws-region>",
    aws_profile="<aws-profile>",
)

embedding = embed_model.get_text_embedding("hello world")

# ## List supported models
# 
# To check list of supported models of Amazon Bedrock on LlamaIndex, call `BedrockEmbedding.list_supported_models()` as follows.

from llama_index.embeddings import BedrockEmbedding
import json

supported_models = BedrockEmbedding.list_supported_models()
print(json.dumps(supported_models, indent=2))

# ## Provider: Amazon
# Amazon Bedrock Titan embeddings.

from llama_index.embeddings import BedrockEmbedding

model = BedrockEmbedding().from_credentials(
    model_name="amazon.titan-embed-g1-text-02"
)
embeddings = model.get_text_embedding("hello world")
print(embeddings)

# ## Provider: Cohere
# 
# ### cohere.embed-english-v3

model = BedrockEmbedding().from_credentials(
    model_name="cohere.embed-english-v3"
)
coherePayload = {
    "texts": ["This is a test document", "This is another test document"],
    "input_type": "search_document",
    "truncate": "NONE",
}
embeddings = model.get_text_embedding(coherePayload)
print(embeddings)

# ### MultiLingual Embeddings from Cohere 

model = BedrockEmbedding().from_credentials(
    model_name="cohere.embed-multilingual-v3"
)
coherePayload = {
    "texts": [
        "This is a test document",
        "తెలుగు అనేది ద్రావిడ భాషల కుటుంబానికి చెందిన భాష.",
    ],
    "input_type": "search_document",
    "truncate": "NONE",
}
embeddings = model.get_text_embedding(coherePayload)
print(embeddings)

