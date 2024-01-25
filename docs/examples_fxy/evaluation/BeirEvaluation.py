#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/evaluation/BeirEvaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # BEIR Out of Domain Benchmark

# About [BEIR](https://github.com/beir-cellar/beir):
# 
# BEIR is a heterogeneous benchmark containing diverse IR tasks. It also provides a common and easy framework for evaluation of your retrieval methods within the benchmark.
# 
# Refer to the repo via the link for a full list of supported datasets.

# Here, we test the `all-MiniLM-L6-v2` sentence-transformer embedding, which is one of the fastest for the given accuracy range. We set the top_k value for the retriever to 30. We also use the nfcorpus dataset.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.evaluation.benchmarks import BeirEvaluator
from llama_index import ServiceContext, VectorStoreIndex

def create_retriever(documents):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context, show_progress=True
    )
    return index.as_retriever(similarity_top_k=30)

BeirEvaluator().run(
    create_retriever, datasets=["nfcorpus"], metrics_k_values=[3, 10, 30]
)

# Higher is better for all the evaluation metrics.
# 
# This [towardsdatascience article](https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54) covers NDCG, MAP and MRR in greater depth.
