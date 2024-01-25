#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/retrievers/auto_merging_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Auto Merging Retriever
# 

# 
# You can define this hierarchy yourself over a set of documents, or you can make use of our brand-new text parser: a HierarchicalNodeParser that takes in a candidate set of documents and outputs an entire hierarchy of nodes, from "coarse-to-fine".

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## Load Data
# 
# Let's first load the Llama 2 paper: https://arxiv.org/pdf/2307.09288.pdf. This will be our test data.

#("mkdir -p 'data/'")
#('wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"')

from pathlib import Path

# from llama_hub.file.pdf.base import PDFReader
from llama_hub.file.pymu_pdf.base import PyMuPDFReader

loader = PyMuPDFReader()
# docs0 = loader.load_data(file=Path("./data/llama2.pdf"))
docs0 = loader.load(file_path=Path("./data/llama2.pdf"))

# By default, the PDF reader creates a separate doc for each page.
# For the sake of this notebook, we stitch docs together into one doc. 
# This will help us better highlight auto-merging capabilities that "stitch" chunks together later on.

from llama_index import Document

doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]

# ## Parse Chunk Hierarchy from Text, Load into Storage
# 

# 
# By default, the hierarchy is:
# - 1st level: chunk size 2048
# - 2nd level: chunk size 512
# - 3rd level: chunk size 128
# 
# 
# We then load these nodes into storage. The leaf nodes are indexed and retrieved via a vector store - these are the nodes that will first be directly retrieved via similarity search. The other nodes will be retrieved from a docstore.

from llama_index.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
)

node_parser = HierarchicalNodeParser.from_defaults()

nodes = node_parser.get_nodes_from_documents(docs)

len(nodes)

# Here we import a simple helper function for fetching "leaf" nodes within a node list. 
# These are nodes that don't have children of their own.

from llama_index.node_parser import get_leaf_nodes, get_root_nodes

leaf_nodes = get_leaf_nodes(nodes)

len(leaf_nodes)

root_nodes = get_root_nodes(nodes)

# ### Load into Storage
# 
# We define a docstore, which we load all nodes into. 
# 
# We then define a `VectorStoreIndex` containing just the leaf-level nodes.

# define storage context
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage import StorageContext
from llama_index import ServiceContext
from llama_index.llms import OpenAI

docstore = SimpleDocumentStore()

# insert nodes into docstore
docstore.add_documents(nodes)

# define storage context (will include vector store by default too)
storage_context = StorageContext.from_defaults(docstore=docstore)

service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo")
)

## Load index into vector index
from llama_index import VectorStoreIndex

base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
    service_context=service_context,
)

# ## Define Retriever

from llama_index.retrievers.auto_merging_retriever import AutoMergingRetriever

base_retriever = base_index.as_retriever(similarity_top_k=6)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)

# query_str = "What were some lessons learned from red-teaming?"
# query_str = "Can you tell me about the key concepts for safety finetuning"
query_str = (
    "What could be the potential outcomes of adjusting the amount of safety"
    " data used in the RLHF stage?"
)

nodes = retriever.retrieve(query_str)
base_nodes = base_retriever.retrieve(query_str)

len(nodes)

len(base_nodes)

from llama_index.response.notebook_utils import #display_source_node

for node in nodes:
    #display_source_node(node, source_length=10000)

for node in base_nodes:
    #display_source_node(node, source_length=10000)

# ## Plug it into Query Engine

from llama_index.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever)
base_query_engine = RetrieverQueryEngine.from_args(base_retriever)

response = query_engine.query(query_str)

print(str(response))

base_response = base_query_engine.query(query_str)

print(str(base_response))

# ## Evaluation
# 
# We evaluate how well the hierarchical retriever works compared to the baseline retriever in a more quantitative manner.
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
# Note: we only generate from the first 20 nodes, since the rest are references
eval_service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4"))
dataset_generator = DatasetGenerator(
    root_nodes[:20],
    service_context=eval_service_context,
    show_progress=True,
    num_questions_per_chunk=3,
)

eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=60)

eval_dataset.save_json("data/llama2_eval_qr_dataset.json")

# optional
eval_dataset = QueryResponseDataset.from_json(
    "data/llama2_eval_qr_dataset.json"
)

# ### Compare Results
# 
# We run evaluations on each of the retrievers: correctness, semantic similarity, relevance, and faithfulness.

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

from collections import defaultdict
import pandas as pd

# NOTE: can uncomment other evaluators
evaluator_c = CorrectnessEvaluator(service_context=eval_service_context)
evaluator_s = SemanticSimilarityEvaluator(service_context=eval_service_context)
evaluator_r = RelevancyEvaluator(service_context=eval_service_context)
evaluator_f = FaithfulnessEvaluator(service_context=eval_service_context)
# pairwise_evaluator = PairwiseComparisonEvaluator(service_context=eval_service_context)

from llama_index.evaluation.eval_utils import get_responses, get_results_df
from llama_index.evaluation import BatchEvalRunner

eval_qs = eval_dataset.questions
qr_pairs = eval_dataset.qr_pairs
ref_response_strs = [r for (_, r) in qr_pairs]

pred_responses = get_responses(eval_qs, query_engine, show_progress=True)

base_pred_responses = get_responses(
    eval_qs, base_query_engine, show_progress=True
)

import numpy as np

pred_response_strs = [str(p) for p in pred_responses]
base_pred_response_strs = [str(p) for p in base_pred_responses]

evaluator_dict = {
    "correctness": evaluator_c,
    "faithfulness": evaluator_f,
    "relevancy": evaluator_r,
    "semantic_similarity": evaluator_s,
}
batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)

eval_results = await batch_runner.aevaluate_responses(
    eval_qs, responses=pred_responses, reference=ref_response_strs
)

base_eval_results = await batch_runner.aevaluate_responses(
    eval_qs, responses=base_pred_responses, reference=ref_response_strs
)

results_df = get_results_df(
    [eval_results, base_eval_results],
    ["Auto Merging Retriever", "Base Retriever"],
    ["correctness", "relevancy", "faithfulness", "semantic_similarity"],
)
#display(results_df)

# **Analysis**: The results are roughly the same.
# 
# Let's also try to see which answer GPT-4 prefers with our pairwise evals.

batch_runner = BatchEvalRunner(
    {"pairwise": pairwise_evaluator}, workers=10, show_progress=True
)

pairwise_eval_results = await batch_runner.aevaluate_response_strs(
    eval_qs,
    response_strs=pred_response_strs,
    reference=base_pred_response_strs,
)
pairwise_score = np.array(
    [r.score for r in pairwise_eval_results["pairwise"]]
).mean()

pairwise_score

# **Analysis**: The pairwise comparison score is a measure of the percentage of time the candidate answer (using auto-merging retriever) is preferred vs. the base answer (using the base retriever). Here we see that it's roughly even.
