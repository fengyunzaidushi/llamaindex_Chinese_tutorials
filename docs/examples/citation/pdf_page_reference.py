#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/citation/pdf_page_reference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    download_loader,
    RAKEKeywordTableIndex,
)

# Set service context to enable streaming

from llama_index import ServiceContext
from llama_index.llms import OpenAI

service_context = ServiceContext.from_defaults(
    llm=OpenAI(temperature=0, model="text-davinci-003")
)

# Download Data

#("mkdir -p 'data/10k/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'")

# Load document and build index

reader = SimpleDirectoryReader(input_files=["./data/10k/lyft_2021.pdf"])
data = reader.load_data()

index = VectorStoreIndex.from_documents(data, service_context=service_context)

query_engine = index.as_query_engine(streaming=True, similarity_top_k=3)

# Stream response with page citation

response = query_engine.query(
    "What was the impact of COVID? Show statements in bullet form and show"
    " page reference after each statement."
)
response.print_response_stream()

for node in response.source_nodes:
    print("-----")
    text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
    print(f"Text:\t {text_fmt} ...")
    print(f"Metadata:\t {node.node.metadata}")
    print(f"Score:\t {node.score:.3f}")

