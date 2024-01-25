#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/output_parsing/GuardrailsDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Guardrails Output Parsing
# 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install guardrails-ai')

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Load documents, build the VectorStoreIndex
# 

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

# #### Define Query + Guardrails Spec
# 

from llama_index.output_parsers import GuardrailsOutputParser

# **Define custom QA and Refine Prompts**
# 

# **Define Guardrails Spec**
# 

# You can either define a RailSpec and initialise a Guard object from_rail_string()
# OR define Pydantic classes and initialise a Guard object from_pydantic()
# For more info: https://docs.guardrailsai.com/defining_guards/pydantic/
# Guardrails recommends Pydantic

from pydantic import BaseModel, Field
from typing import List
import guardrails as gd

class Point(BaseModel):
    
    # Left out for brevity
    explanation: str = Field()
    explanation2: str = Field()
    explanation3: str = Field()

class BulletPoints(BaseModel):
    points: List[Point] = Field(
        description="Bullet points regarding events in the author's life."
    )

# Define the prompt
prompt = """
Query string here.

${gr.xml_prefix_prompt}

${output_schema}

${gr.json_suffix_prompt_v2_wo_none}
"""

from llama_index.llms import OpenAI

# Create a guard object
guard = gd.Guard.from_pydantic(output_class=BulletPoints, prompt=prompt)

# Create output parse object
output_parser = GuardrailsOutputParser(guard, llm=OpenAI())

# attach to an llm object
llm = OpenAI(output_parser=output_parser)

from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
)

# take a look at the new QA template!
fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
print(fmt_qa_tmpl)

# #### Query Index
# 

from llama_index import ServiceContext

ctx = ServiceContext.from_defaults(llm=llm)

query_engine = index.as_query_engine(
    service_context=ctx,
)
response = query_engine.query(
    "What are the three items the author did growing up?",
)

print(response)

