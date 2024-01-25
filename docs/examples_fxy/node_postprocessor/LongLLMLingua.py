#!/usr/bin/env python
# coding: utf-8

# # LongLLMLingua
# 
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/node_postprocessor/LongLLMLingua.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# LongLLMLingua is a research project/paper that presents a new method for prompt compression in the long-context setting.
# 
# - Paper: https://arxiv.org/abs/2310.06839
# - Repo: https://github.com/microsoft/LLMLingua
# 

# 
# **NOTE**: we don't implement the [subsequence recovery method](https://github.com/microsoft/LLMLingua/blob/main/DOCUMENT.md#post-precessing) since that is after the step of processing the nodes.
# 
# **NOTE**: You need quite a bit of RAM/GPU capacity to run this. We got it working on Colab Pro with a V100 instance.

#('pip install llmlingua llama-index')

import openai

openai.api_key = "<insert_openai_key>"

# ## Setup (Data + Index)
# 
# We load in PG's essay, index it, and define a retriever.

#('wget "https://www.dropbox.com/s/f6bmb19xdg0xedm/paul_graham_essay.txt?dl=1" -O paul_graham_essay.txt')

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
)

# load documents
documents = SimpleDirectoryReader(
    input_files=["paul_graham_essay.txt"]
).load_data()

index = VectorStoreIndex.from_documents(documents)

retriever = index.as_retriever(similarity_top_k=2)

# query_str = "What did the author do growing up?"
# query_str = "What did the author do during his time in YC?"
query_str = "Where did the author go for art school?"

results = retriever.retrieve(query_str)
print(results)

results

# ## Setup LongLLMLingua as a Postprocessor
# 
# We setup `LongLLMLinguaPostprocessor` which will use the `longllmlingua` package to run prompt compression.
# 
# We specify a target token size of 300, and supply an instruction string.
# 
# Special thanks to Huiqiang J. for the help with the parameters.

from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import CompactAndRefine
from llama_index.postprocessor import LongLLMLinguaPostprocessor

node_postprocessor = LongLLMLinguaPostprocessor(
    instruction_str="Given the context, please answer the final question",
    target_token=300,
    rank_method="longllmlingua",
    additional_compress_kwargs={
        "condition_compare": True,
        "condition_in_question": "after",
        "context_budget": "+100",
        "reorder_context": "sort",  # enable document reorder
    },
)

# ## Try It Out
# 
# We show you how to compose a retriever + compressor + query engine into a RAG pipeline.
# 1. We show you this step by step.
# 2. We show you how to do this in an out-of-the-box fashion with our `RetrieverQueryEngine`.

# ### Step-by-Step

retrieved_nodes = retriever.retrieve(query_str)
synthesizer = CompactAndRefine()

from llama_index.schema import QueryBundle

# outline steps in RetrieverQueryEngine for clarity:
# postprocess (compress), synthesize
new_retrieved_nodes = node_postprocessor.postprocess_nodes(
    retrieved_nodes, query_bundle=QueryBundle(query_str=query_str)
)

print("\n\n".join([n.get_content() for n in new_retrieved_nodes]))

response = synthesizer.synthesize(query_str, new_retrieved_nodes)

print(str(response))

# ### Out of the box with `RetrieverQueryEngine`

retriever_query_engine = RetrieverQueryEngine.from_args(
    retriever, node_postprocessors=[node_postprocessor]
)

response = retriever_query_engine.query(query_str)
print(str(response))

