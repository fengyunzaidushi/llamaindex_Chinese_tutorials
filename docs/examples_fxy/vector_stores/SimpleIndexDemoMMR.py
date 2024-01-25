#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/SimpleIndexDemoMMR.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Simple Vector Stores - Maximum Marginal Relevance Retrieval

# This notebook explores the use of MMR retrieval [<a href="https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf">1</a>]. By using maximum marginal relevance, one can iteratively find documents that are dissimilar to previous results. It has been shown to improve performance for LLM retrievals [<a href="https://arxiv.org/pdf/2211.13892.pdf">2</a>]. 
# 
# The maximum marginal relevance algorithm is as follows:
# $$
# \text{{MMR}} = \arg\max_{d_i \in D \setminus R} [ \lambda \cdot Sim_1(d_i, q) - (1 - \lambda) \cdot \max_{d_j \in R} Sim_2(d_i, d_j) ]
# $$
# 
# Here, D is the set of all candidate documents, R is the set of already selected documents, q is the query, $Sim_1$ is the similarity function between a document and the query, and $Sim_2$ is the similarity function between two documents. $d_i$ and $d_j$ are documents in D and R respectively.
# 
# The parameter λ (mmr_threshold) controls the trade-off between relevance (the first term) and diversity (the second term). If mmr_threshold is close to 1, more emphasis is put on relevance, while a mmr_threshold close to 0 puts more emphasis on diversity.

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

from llama_index import VectorStoreIndex, SimpleDirectoryReader

# llama_index/docs/examples/data/paul_graham
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents)

# To use mmr, set it as a vector_store_query_mode
query_engine = index.as_query_engine(vector_store_query_mode="mmr")
response = query_engine.query("What did the author do growing up?")
print(response)

from llama_index import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents)

# To set the threshold, set it in vector_store_kwargs
query_engine_with_threshold = index.as_query_engine(
    vector_store_query_mode="mmr", vector_store_kwargs={"mmr_threshold": 0.2}
)

response = query_engine_with_threshold.query(
    "What did the author do growing up?"
)
print(response)

# Note that the node score will be scaled with the threshold and will additionally be penalized for the similarity to previous nodes. As the threshold goes to 1, the scores will become equal and similarity to previous nodes will be ignored, turning off the impact of MMR. By lowering the threshold, the algorithm will prefer more diverse documents.

index1 = VectorStoreIndex.from_documents(documents)
query_engine_no_mrr = index1.as_query_engine()
response_no_mmr = query_engine_no_mrr.query(
    "What did the author do growing up?"
)

index2 = VectorStoreIndex.from_documents(documents)
query_engine_with_high_threshold = index2.as_query_engine(
    vector_store_query_mode="mmr", vector_store_kwargs={"mmr_threshold": 0.8}
)
response_low_threshold = query_engine_with_low_threshold.query(
    "What did the author do growing up?"
)

index3 = VectorStoreIndex.from_documents(documents)
query_engine_with_low_threshold = index3.as_query_engine(
    vector_store_query_mode="mmr", vector_store_kwargs={"mmr_threshold": 0.2}
)
response_high_threshold = query_engine_with_high_threshold.query(
    "What did the author do growing up?"
)

print(
    "Scores without MMR ",
    [node.score for node in response_no_mmr.source_nodes],
)
print(
    "Scores with MMR and a threshold of 0.8 ",
    [node.score for node in response_high_threshold.source_nodes],
)
print(
    "Scores with MMR and a threshold of 0.2 ",
    [node.score for node in response_low_threshold.source_nodes],
)

# ## Retrieval-Only Demonstration
# 
# By setting a small chunk size and adjusting the "mmr_threshold" parameter, we can see how the retrieved results
# change from very diverse (and less relevant) to less diverse (and more relevant/redundant).
# 
# We try the following values: 0.1, 0.5, 0.8, 1.0

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.response.notebook_utils import #display_source_node
from llama_index.llms import OpenAI

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=64)

# llama_index/docs/examples/data/paul_graham
documents = SimpleDirectoryReader("../data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

retriever = index.as_retriever(
    vector_store_query_mode="mmr",
    similarity_top_k=3,
    vector_store_kwargs={"mmr_threshold": 0.1},
)
nodes = retriever.retrieve(
    "What did the author do during his time in Y Combinator?"
)

for n in nodes:
    #display_source_node(n, source_length=1000)

retriever = index.as_retriever(
    vector_store_query_mode="mmr",
    similarity_top_k=3,
    vector_store_kwargs={"mmr_threshold": 0.5},
)
nodes = retriever.retrieve(
    "What did the author do during his time in Y Combinator?"
)

for n in nodes:
    #display_source_node(n, source_length=1000)

retriever = index.as_retriever(
    vector_store_query_mode="mmr",
    similarity_top_k=3,
    vector_store_kwargs={"mmr_threshold": 0.8},
)
nodes = retriever.retrieve(
    "What did the author do during his time in Y Combinator?"
)

for n in nodes:
    #display_source_node(n, source_length=1000)

retriever = index.as_retriever(
    vector_store_query_mode="mmr",
    similarity_top_k=3,
    vector_store_kwargs={"mmr_threshold": 1.0},
)
nodes = retriever.retrieve(
    "What did the author do during his time in Y Combinator?"
)

for n in nodes:
    #display_source_node(n, source_length=1000)

