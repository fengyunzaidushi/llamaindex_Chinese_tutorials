#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/retrievers/ensemble_retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Ensemble Retrieval Guide
# 
# Oftentimes when building a RAG applications there are many retreival parameters/strategies to decide from (from chunk size to vector vs. keyword vs. hybrid search, for instance).
# 
# Thought: what if we could try a bunch of strategies at once, and have any AI/reranker/LLM prune the results?
# 
# This achieves two purposes:
# - Better (albeit more costly) retrieved results by pooling results from multiple strategies, assuming the reranker is good
# - A way to benchmark different retrieval strategies against each other (w.r.t reranker)
# 
# This guide showcases this over the Llama 2 paper. We do ensemble retrieval over different chunk sizes and also different indices.
# 
# **NOTE**: A closely related guide is our [Ensemble Query Engine Guide](https://gpt-index.readthedocs.io/en/stable/examples/query_engine/ensemble_qury_engine.html) - make sure to check it out! 

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# ## Setup
# 
# Here we define the necessary imports.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# NOTE: This is ONLY necessary in jupyter notebook.
# Details: Jupyter runs an event-loop behind the scenes.
#          This results in nested event-loops when we start an event-loop to make async queries.
#          This is normally not allowed, we use nest_asyncio to allow it for convenience.
import nest_asyncio

nest_asyncio.apply()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)
from llama_index.response.notebook_utils import #display_response
from llama_index.llms import OpenAI

# ## Load Data
# 

#('wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"')

from pathlib import Path
from llama_index import Document
from llama_hub.file.pymu_pdf.base import PyMuPDFReader

loader = PyMuPDFReader()
docs0 = loader.load(file_path=Path("./data/llama2.pdf"))
doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]

# Here we try out different chunk sizes: 128, 256, 512, and 1024.

# initialize service context (set chunk size)
llm = OpenAI(model="gpt-4")
chunk_sizes = [128, 256, 512, 1024]
service_contexts = []
nodes_list = []
vector_indices = []
query_engines = []
for chunk_size in chunk_sizes:
    print(f"Chunk Size: {chunk_size}")
    service_context = ServiceContext.from_defaults(
        chunk_size=chunk_size, llm=llm
    )
    service_contexts.append(service_context)
    nodes = service_context.node_parser.get_nodes_from_documents(docs)

    # add chunk size to nodes to track later
    for node in nodes:
        node.metadata["chunk_size"] = chunk_size
        node.excluded_embed_metadata_keys = ["chunk_size"]
        node.excluded_llm_metadata_keys = ["chunk_size"]

    nodes_list.append(nodes)

    # build vector index
    vector_index = VectorStoreIndex(nodes)
    vector_indices.append(vector_index)

    # query engines
    query_engines.append(vector_index.as_query_engine())

# ## Define Ensemble Retriever
# 
# We setup an "ensemble" retriever primarily using our recursive retrieval abstraction. This works like the following:
# - Define a separate `IndexNode` corresponding to the vector retriever for each chunk size (retriever for chunk size 128, retriever for chunk size 256, and more)
# - Put all IndexNodes into a single `SummaryIndex` - when the corresponding retriever is called, *all* nodes are returned.
# - Define a Recursive Retriever, with the root node being the summary index retriever. This will first fetch all nodes from the summary index retriever, and then recursively call the vector retriever for each chunk size.
# - Rerank the final results.
# 
# The end result is that all vector retrievers are called when a query is run.

# try ensemble retrieval

from llama_index.tools import RetrieverTool
from llama_index.schema import IndexNode

# retriever_tools = []
retriever_dict = {}
retriever_nodes = []
for chunk_size, vector_index in zip(chunk_sizes, vector_indices):
    node_id = f"chunk_{chunk_size}"
    node = IndexNode(
        text=(
            "Retrieves relevant context from the Llama 2 paper (chunk size"
            f" {chunk_size})"
        ),
        index_id=node_id,
    )
    retriever_nodes.append(node)
    retriever_dict[node_id] = vector_index.as_retriever()

# Define recursive retriever.

from llama_index.selectors.pydantic_selectors import PydanticMultiSelector

# from llama_index.retrievers import RouterRetriever
from llama_index.retrievers import RecursiveRetriever
from llama_index import SummaryIndex

# the derived retriever will just retrieve all nodes
summary_index = SummaryIndex(retriever_nodes)

retriever = RecursiveRetriever(
    root_id="root",
    retriever_dict={"root": summary_index.as_retriever(), **retriever_dict},
)

# Let's test the retriever on a sample query.

nodes = await retriever.aretrieve(
    "Tell me about the main aspects of safety fine-tuning"
)

print(f"Number of nodes: {len(nodes)}")
for node in nodes:
    print(node.node.metadata["chunk_size"])
    print(node.node.get_text())

# Define reranker to process the final retrieved set of nodes.

# define reranker
from llama_index.postprocessor import (
    LLMRerank,
    SentenceTransformerRerank,
    CohereRerank,
)

# reranker = LLMRerank()
# reranker = SentenceTransformerRerank(top_n=10)
reranker = CohereRerank(top_n=10)

# Define retriever query engine to integrate the recursive retriever + reranker together.

# define RetrieverQueryEngine
from llama_index.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine(retriever, node_postprocessors=[reranker])

response = query_engine.query(
    "Tell me about the main aspects of safety fine-tuning"
)

#display_response(
    response, show_source=True, source_length=500, show_source_metadata=True
)

# ### Analyzing the Relative Importance of each Chunk
# 
# One interesting property of ensemble-based retrieval is that through reranking, we can actually use the ordering of chunks in the final retrieved set to determine the importance of each chunk size. For instance, if certain chunk sizes are always ranked near the top, then those are probably more relevant to the query.

# compute the average precision for each chunk size based on positioning in combined ranking
from collections import defaultdict
import pandas as pd

def mrr_all(metadata_values, metadata_key, source_nodes):
    # source nodes is a ranked list
    # go through each value, find out positioning in source_nodes
    value_to_mrr_dict = {}
    for metadata_value in metadata_values:
        mrr = 0
        for idx, source_node in enumerate(source_nodes):
            if source_node.node.metadata[metadata_key] == metadata_value:
                mrr = 1 / (idx + 1)
                break
            else:
                continue

        # normalize AP, set in dict
        value_to_mrr_dict[metadata_value] = mrr

    df = pd.DataFrame(value_to_mrr_dict, index=["MRR"])
    df.style.set_caption("Mean Reciprocal Rank")
    return df

# Compute the Mean Reciprocal Rank for each chunk size (higher is better)
# we can see that chunk size of 256 has the highest ranked results.
print("Mean Reciprocal Rank for each Chunk Size")
mrr_all(chunk_sizes, "chunk_size", response.source_nodes)

# ## Evaluation
# 
# We more rigorously evaluate how well an ensemble retriever works compared to the "baseline" retriever.
# 
# We define/load an eval benchmark dataset and then run different evaluations over it.
# 
# **WARNING**: This can be *expensive*, especially with GPT-4. Use caution and tune the sample size to fit your budget.

from llama_index.evaluation import (
    DatasetGenerator,
    QueryResponseDataset,
)
from llama_index import ServiceContext
from llama_index.llms import OpenAI
import nest_asyncio

nest_asyncio.apply()

# NOTE: run this if the dataset isn't already saved
eval_service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4"))
# generate questions from the largest chunks (1024)
dataset_generator = DatasetGenerator(
    nodes_list[-1],
    service_context=eval_service_context,
    show_progress=True,
    num_questions_per_chunk=2,
)

eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=60)

eval_dataset.save_json("data/llama2_eval_qr_dataset.json")

# optional
eval_dataset = QueryResponseDataset.from_json(
    "data/llama2_eval_qr_dataset.json"
)

# ### Compare Results

import asyncio
import nest_asyncio

nest_asyncio.apply()

from llama_index.evaluation import (
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    PairwiseComparisonEvaluator,
)

# NOTE: can uncomment other evaluators
evaluator_c = CorrectnessEvaluator(service_context=eval_service_context)
evaluator_s = SemanticSimilarityEvaluator(service_context=eval_service_context)
evaluator_r = RelevancyEvaluator(service_context=eval_service_context)
evaluator_f = FaithfulnessEvaluator(service_context=eval_service_context)

pairwise_evaluator = PairwiseComparisonEvaluator(
    service_context=eval_service_context
)

from llama_index.evaluation.eval_utils import get_responses, get_results_df
from llama_index.evaluation import BatchEvalRunner

max_samples = 60

eval_qs = eval_dataset.questions
qr_pairs = eval_dataset.qr_pairs
ref_response_strs = [r for (_, r) in qr_pairs]

# resetup base query engine and ensemble query engine
# base query engine
base_query_engine = vector_indices[-1].as_query_engine(similarity_top_k=2)
# ensemble query engine
reranker = CohereRerank(top_n=4)
query_engine = RetrieverQueryEngine(retriever, node_postprocessors=[reranker])

base_pred_responses = get_responses(
    eval_qs[:max_samples], base_query_engine, show_progress=True
)

pred_responses = get_responses(
    eval_qs[:max_samples], query_engine, show_progress=True
)

import numpy as np

pred_response_strs = [str(p) for p in pred_responses]
base_pred_response_strs = [str(p) for p in base_pred_responses]

evaluator_dict = {
    "correctness": evaluator_c,
    "faithfulness": evaluator_f,
    # "relevancy": evaluator_r,
    "semantic_similarity": evaluator_s,
}
batch_runner = BatchEvalRunner(evaluator_dict, workers=1, show_progress=True)

eval_results = await batch_runner.aevaluate_responses(
    queries=eval_qs[:max_samples],
    responses=pred_responses[:max_samples],
    reference=ref_response_strs[:max_samples],
)

base_eval_results = await batch_runner.aevaluate_responses(
    queries=eval_qs[:max_samples],
    responses=base_pred_responses[:max_samples],
    reference=ref_response_strs[:max_samples],
)

results_df = get_results_df(
    [eval_results, base_eval_results],
    ["Ensemble Retriever", "Base Retriever"],
    ["correctness", "faithfulness", "semantic_similarity"],
)
#display(results_df)

batch_runner = BatchEvalRunner(
    {"pairwise": pairwise_evaluator}, workers=3, show_progress=True
)

pairwise_eval_results = await batch_runner.aevaluate_response_strs(
    queries=eval_qs[:max_samples],
    response_strs=pred_response_strs[:max_samples],
    reference=base_pred_response_strs[:max_samples],
)

results_df = get_results_df(
    [eval_results, base_eval_results],
    ["Ensemble Retriever", "Base Retriever"],
    ["pairwise"],
)
#display(results_df)

