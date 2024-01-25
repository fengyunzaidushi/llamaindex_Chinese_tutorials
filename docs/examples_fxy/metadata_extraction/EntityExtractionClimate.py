#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/metadata_extraction/EntityExtractionClimate.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Entity Metadata Extraction
# 

# 
# For more information on metadata extraction in LlamaIndex, see our [documentation](https://gpt-index.readthedocs.io/en/stable/core_modules/data_modules/documents_and_nodes/usage_metadata_extractor.html).

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# Needed to run the entity extractor
# !pip install span_marker

import os
import openai

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

# ## Setup the Extractor and Parser

from llama_index.extractors.metadata_extractors import EntityExtractor
from llama_index.node_parser import SentenceSplitter

entity_extractor = EntityExtractor(
    prediction_threshold=0.5,
    label_entities=False,  # include the entity label in the metadata (can be erroneous)
    device="cpu",  # set to "cuda" if you have a GPU
)

node_parser = SentenceSplitter()

transformations = [node_parser, entity_extractor]

# ## Load the data
# 
# Here, we will download the 2023 IPPC Climate Report - Chapter 3 on Oceans and Coastal Ecosystems (172 Pages)

#('curl https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter03.pdf --output IPCC_AR6_WGII_Chapter03.pdf')

# Next, load the documents.

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# ## Extracting Metadata
# 
# Now, this is a pretty long document. Since we are not running on CPU, for now, we will only run on a subset of documents. Feel free to run it on all documents on your own though!

from llama_index.ingestion import IngestionPipeline

import random

random.seed(42)
# comment out to run on all documents
# 100 documents takes about 5 minutes on CPU
documents = random.sample(documents, 100)

pipeline = IngestionPipeline(transformations=transformations)

nodes = pipeline.run(documents=documents)

# ### Examine the outputs

samples = random.sample(nodes, 5)
for node in samples:
    print(node.metadata)

# ## Try a Query!

from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI

service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2)
)

index = VectorStoreIndex(nodes, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("What is said by Fox-Kemper?")
print(response)

# ### Contrast without metadata
# 
# Here, we re-construct the index, but without metadata

for node in nodes:
    node.metadata.pop("entities", None)

print(nodes[0].metadata)

from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI

service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2)
)

index = VectorStoreIndex(nodes, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("What is said by Fox-Kemper?")
print(response)

# As we can see, our metadata-enriched index is able to fetch more relevant information.
