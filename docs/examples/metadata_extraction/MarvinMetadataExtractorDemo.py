#!/usr/bin/env python
# coding: utf-8

# # Metadata Extraction and Augmentation w/ Marvin
# 
# This notebook walks through using [`Marvin`](https://github.com/PrefectHQ/marvin) to extract and augment metadata from text. Marvin uses the LLM to identify and extract metadata.  Metadata can be anything from additional and enhanced questions and answers to business object identification and elaboration.  This notebook will demonstrate pulling out and elaborating on Sports Supplement information in a csv document.
# 
# Note: You will need to supply a valid open ai key below to run this notebook.

# ## Setup

# !pip install marvin

from llama_index import SimpleDirectoryReader
from llama_index.indices.service_context import ServiceContext
from llama_index.llms import OpenAI
from llama_index.node_parser import TokenTextSplitter
from llama_index.extractors.marvin_metadata_extractor import (
    MarvinMetadataExtractor,
)

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

documents = SimpleDirectoryReader("data").load_data()

# limit document text length
documents[0].text = documents[0].text[:10000]

import marvin
from marvin import ai_model

from llama_index.bridge.pydantic import BaseModel, Field

marvin.settings.openai.api_key = os.environ["OPENAI_API_KEY"]

@ai_model
class SportsSupplement(BaseModel):
    name: str = Field(..., description="The name of the sports supplement")
    description: str = Field(
        ..., description="A description of the sports supplement"
    )
    pros_cons: str = Field(
        ..., description="The pros and cons of the sports supplement"
    )

llm_model = "gpt-3.5-turbo"

llm = OpenAI(temperature=0.1, model_name=llm_model, max_tokens=512)
service_context = ServiceContext.from_defaults(llm=llm)

# construct text splitter to split texts into chunks for processing
# this takes a while to process, you can increase processing time by using larger chunk_size
# file size is a factor too of course
node_parser = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128
)

# set the global service context object, avoiding passing service_context when building the index
from llama_index import set_global_service_context

set_global_service_context(service_context)

# create metadata extractor
metadata_extractor = MarvinMetadataExtractor(
    marvin_model=SportsSupplement, llm_model_string=llm_model
)  # let's extract custom entities for each node.

# use node_parser to get nodes from the documents
from llama_index.ingestion import IngestionPipeline

pipeline = IngestionPipeline(transformations=[node_parser, metadata_extractor])

nodes = pipeline.run(documents=documents, show_progress=True)

from pprint import pprint

for i in range(5):
    pprint(nodes[i].metadata)

