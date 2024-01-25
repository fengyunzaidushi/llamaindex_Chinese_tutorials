#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/customization/llms/AzureOpenAI.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Azure OpenAI

# Azure openAI resources unfortunately differ from standard openAI resources as you can't generate embeddings unless you use an embedding model. The regions where these models are available can be found here: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/models#embeddings-models
# 
# Furthermore the regions that support embedding models unfortunately don't support the latest versions (<*>-003) of openAI models, so we are forced to use one region for embeddings and another for the text generation.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Here, we setup the embedding model (for retrieval) and llm (for text generation).
# Note that you need not only model names (e.g. "text-embedding-ada-002"), but also model deployment names (the one you chose when deploying the model in Azure.
# You must pass the deployment name as a parameter when you initialize `AzureOpenAI` and `OpenAIEmbedding`.

api_key = "<api-key>"
azure_endpoint = "https://<your-resource-name>.openai.azure.com/"
api_version = "2023-07-01-preview"

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="my-custom-llm",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="my-custom-embedding",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

from llama_index import set_global_service_context

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

set_global_service_context(service_context)

documents = SimpleDirectoryReader(
    input_files=["../../data/paul_graham/paul_graham_essay.txt"]
).load_data()
index = VectorStoreIndex.from_documents(documents)

query = "What is most interesting about this essay?"
query_engine = index.as_query_engine()
answer = query_engine.query(query)

print(answer.get_formatted_sources())
print("query was:", query)
print("answer was:", answer)

