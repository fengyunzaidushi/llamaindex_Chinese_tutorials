#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/output_parsing/LangchainOutputParserDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Langchain Output Parsing

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Load documents, build the VectorStoreIndex

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from IPython.#display import Markdown, #display

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

index = VectorStoreIndex.from_documents(documents, chunk_size=512)

# #### Define Query + Langchain Output Parser

from llama_index.output_parsers import LangchainOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# **Define custom QA and Refine Prompts**

response_schemas = [
    ResponseSchema(
        name="Education",
        description=(
            "Describes the author's educational experience/background."
        ),
    ),
    ResponseSchema(
        name="Work",
        description="Describes the author's work experience/background.",
    ),
]

lc_output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)
output_parser = LangchainOutputParser(lc_output_parser)

from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
)

# take a look at the new QA template!
fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
print(fmt_qa_tmpl)

# #### Query Index

from llama_index import ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(output_parser=output_parser)
ctx = ServiceContext.from_defaults(llm=llm)

query_engine = index.as_query_engine(
    service_context=ctx,
)
response = query_engine.query(
    "What are a few things the author did growing up?",
)

print(response)

