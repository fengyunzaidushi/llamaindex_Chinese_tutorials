#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/low_level/fusion_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Building an Advanced Fusion Retriever from Scratch
# 

# 
# Specifically, we show you how to build our `QueryFusionRetriever` from scratch.
# 
# This is heavily inspired from the RAG-fusion repo here: https://github.com/Raudaschl/rag-fusion.

# ## Setup
# 
# We load documents and build a simple vector index.

#('pip install rank-bm25 pymupdf')

import nest_asyncio

nest_asyncio.apply()

# #### Load Documents

#('mkdir data')
#('wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"')

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from pathlib import Path
from llama_hub.file.pymu_pdf.base import PyMuPDFReader

loader = PyMuPDFReader()
documents = loader.load(file_path="./data/llama2.pdf")

# #### Load into Vector Store

from llama_index import VectorStoreIndex, ServiceContext

service_context = ServiceContext.from_defaults(chunk_size=1024)
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# #### Define LLMs

from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

# ## Define Advanced Retriever
# 
# We define an advanced retriever that performs the following steps:
# 1. Query generation/rewriting: generate multiple queries given the original user query
# 2. Perform retrieval for each query over an ensemble of retrievers.
# 3. Reranking/fusion: fuse results from all queries, and apply a reranking step to "fuse" the top relevant results!
# 
# Then in the next section we'll plug this into our response synthesis module.

# ### Step 1: Query Generation/Rewriting
# 
# The first step is to generate queries from the original query to better match the query intent, and increase precision/recall of the retrieved results. For instance, we might be able to rewrite the query into smaller queries.
# 
# We can do this by prompting ChatGPT.

from llama_index import PromptTemplate

query_str = "How do the models developed in this work compare to open-source chat models based on the benchmarks tested?"

query_gen_prompt_str = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)
query_gen_prompt = PromptTemplate(query_gen_prompt_str)

def generate_queries(llm, query_str: str, num_queries: int = 4):
    fmt_prompt = query_gen_prompt.format(
        num_queries=num_queries - 1, query=query_str
    )
    response = llm.complete(fmt_prompt)
    queries = response.text.split("\n")
    return queries

queries = generate_queries(llm, query_str, num_queries=4)

print(queries)

# ### Step 2: Perform Vector Search for Each Query
# 
# Now we run retrieval for each query. This means that we fetch the top-k most relevant results from each vector store.
# 
# **NOTE**: We can also have multiple retrievers. Then the total number of queries we run is N*M, where N is number of retrievers and M is number of generated queries. Hence there will also be N*M retrieved lists.
# 
# Here we'll use the retriever provided from our vector store. If you want to see how to build this from scratch please see [our tutorial on this](https://docs.llamaindex.ai/en/latest/examples/low_level/retrieval.html#put-this-into-a-retriever).

from tqdm.asyncio import tqdm

async def run_queries(queries, retrievers):
    """Run queries against retrievers."""
    tasks = []
    for query in queries:
        for i, retriever in enumerate(retrievers):
            tasks.append(retriever.aretrieve(query))

    task_results = await tqdm.gather(*tasks)

    results_dict = {}
    for i, (query, query_result) in enumerate(zip(queries, task_results)):
        results_dict[(query, i)] = query_result

    return results_dict

# get retrievers
from llama_index.retrievers import BM25Retriever

## vector retriever
vector_retriever = index.as_retriever(similarity_top_k=2)

## bm25 retriever
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=2
)

results_dict = await run_queries(queries, [vector_retriever, bm25_retriever])

# ### Step 3: Perform Fusion
# 
# The next step here is to perform fusion: combining the results from several retrievers into one and re-ranking.
# 
# Note that a given node might be retrieved multiple times from different retrievers, so there needs to be a way to de-dup and rerank the node given the multiple retrievals.
# 
# We'll show you how to perform "reciprocal rank fusion": for each node, add up its reciprocal rank in every list where it's retrieved.
# 
# Then reorder nodes by highest score to least.
# 
# Full paper here: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

def fuse_results(results_dict, similarity_top_k: int = 2):
    """Fuse results."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
            sorted(
                nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True
            )
        ):
            text = node_with_score.node.get_content()
            text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes[:similarity_top_k]

final_results = fuse_results(results_dict)

from llama_index.response.notebook_utils import #display_source_node

for n in final_results:
    #display_source_node(n, source_length=500)

# **Analysis**: The above code has a few straightforward components.
# 1. Go through each node in each retrieved list, and add it's reciprocal rank to the node's ID. The node's ID is the hash of it's text for dedup purposes.
# 2. Sort results by highest-score to lowest.
# 3. Adjust node scores.

# ## Plug into RetrieverQueryEngine
# 
# Now we're ready to define this as a custom retriever, and plug it into our `RetrieverQueryEngine` (which does retrieval and synthesis).

from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever
from typing import Any, List
from llama_index.schema import NodeWithScore

class FusionRetriever(BaseRetriever):
    """Ensemble retriever with fusion."""

    def __init__(
        self,
        llm,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        queries = generate_queries(llm, query_str, num_queries=4)
        results = run_queries(queries, [vector_retriever, bm25_retriever])
        final_results = fuse_results(
            results_dict, similarity_top_k=self._similarity_top_k
        )

        return final_results

from llama_index.query_engine import RetrieverQueryEngine

fusion_retriever = FusionRetriever(
    llm, [vector_retriever, bm25_retriever], similarity_top_k=2
)

query_engine = RetrieverQueryEngine(fusion_retriever)

response = query_engine.query(query_str)

print(str(response))

