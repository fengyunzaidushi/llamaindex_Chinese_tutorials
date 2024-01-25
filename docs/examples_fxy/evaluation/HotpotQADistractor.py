#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/evaluation/HotpotQADistractor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # HotpotQADistractor Demo
# 
# This notebook walks through evaluating a query engine using the HotpotQA dataset. In this task, the LLM must answer a question given a pre-configured context. The answer usually has to be concise, and accuracy is measured by calculating the overlap (measured by F1) and exact match.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.evaluation.benchmarks import HotpotQAEvaluator
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.schema import Document
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

service_context = ServiceContext.from_defaults(
    embed_model="local:sentence-transformers/all-MiniLM-L6-v2",
    llm=llm,
)
index = VectorStoreIndex.from_documents(
    [Document.example()], service_context=service_context, show_progress=True
)

# First we try with a very simple engine. In this particular benchmark, the retriever and hence index is actually ignored, as the documents retrieved for each query is provided in the dataset. This is known as the "distractor" setting in HotpotQA.

engine = index.as_query_engine(service_context=service_context)

HotpotQAEvaluator().run(engine, queries=5, show_result=True)

# Now we try with a sentence transformer reranker, which selects 3 out of the 10 nodes proposed by the retriever

from llama_index.postprocessor import SentenceTransformerRerank

rerank = SentenceTransformerRerank(top_n=3)

engine = index.as_query_engine(
    service_context=service_context,
    node_postprocessors=[rerank],
)

HotpotQAEvaluator().run(engine, queries=5, show_result=True)

# The F1 and exact match scores appear to improve slightly.
# 
# Note that the benchmark optimizes for producing short factoid answers without explanations, although it is known that CoT prompting can sometimes help in output quality. 
# 
# The scores used are also not a perfect measure of correctness, but can be a quick way to identify how changes in your query engine change the output.
