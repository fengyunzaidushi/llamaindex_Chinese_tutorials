#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/customization/llms/SimpleIndexDemo-Huggingface_camel.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # HuggingFace LLM - Camel-5b

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Load documents, build the VectorStoreIndex

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# setup prompts - specific to StableLM
from llama_index.prompts import PromptTemplate

# This will wrap the default prompts that are internal to llama-index
# taken from https://huggingface.co/Writer/camel-5b-hf
query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "##
)

import torch

llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.25, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="Writer/camel-5b-hf",
    model_name="Writer/camel-5b-hf",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16}
)
service_context = ServiceContext.from_defaults(chunk_size=512, llm=llm)

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# #### Query Index

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

print(response)

# #### Query Index - Streaming

query_engine = index.as_query_engine(streaming=True)

# set Logging to DEBUG for more detailed outputs
response_stream = query_engine.query("What did the author do growing up?")

# can be slower to start streaming since llama-index often involves many LLM calls
response_stream.print_response_stream()

# can also get a normal response object
response = response_stream.get_response()
print(response)

# can also iterate over the generator yourself
generated_text = ""
for text in response.response_gen:
    generated_text += text

