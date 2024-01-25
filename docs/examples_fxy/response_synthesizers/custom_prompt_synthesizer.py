#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/response_synthesizers/custom_prompt_synthesizer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Pydantic Tree Summarize
# 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ## Load Data

from llama_index import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_files=["./data/paul_graham/paul_graham_essay.txt"]
)

docs = reader.load_data()

text = docs[0].text

# ## Define Custom Prompt

from llama_index import PromptTemplate

# NOTE: we add an extra tone_name variable here
qa_prompt_tmpl = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Please also write the answer in the style of {tone_name}.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt = PromptTemplate(qa_prompt_tmpl)

refine_prompt_tmpl = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the query. "
    "Please also write the answer in the style of {tone_name}.\n"
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)
refine_prompt = PromptTemplate(refine_prompt_tmpl)

# ## Try out Response Synthesis with Custom Prompt
# 
# We try out a few different response synthesis strategies with the custom prompt.

from llama_index.response_synthesizers import TreeSummarize, Refine
from llama_index.types import BaseModel
from typing import List

summarizer = TreeSummarize(verbose=True, summary_template=qa_prompt)

response = summarizer.get_response(
    "who is Paul Graham?", [text], tone_name="a Shakespeare play"
)

print(str(response))

summarizer = Refine(
    verbose=True, text_qa_template=qa_prompt, refine_template=refine_prompt
)

response = summarizer.get_response(
    "who is Paul Graham?", [text], tone_name="a haiku"
)

print(str(response))

# try with pydantic model
class Biography(BaseModel):
    """Data model for a biography."""

    name: str
    best_known_for: List[str]
    extra_info: str

summarizer = TreeSummarize(
    verbose=True, summary_template=qa_prompt, output_cls=Biography
)

response = summarizer.get_response(
    "who is Paul Graham?", [text], tone_name="a business memo"
)

print(str(response))

