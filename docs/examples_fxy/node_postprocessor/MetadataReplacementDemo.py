#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/node_postprocessor/MetadataReplacementDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Metadata Replacement + Node Sentence Window
# 

# 
# Then, during retrieval, before passing the retrieved sentences to the LLM, the single sentences are replaced with a window containing the surrounding sentences using the `MetadataReplacementNodePostProcessor`.
# 
# This is most useful for large documents/indexes, as it helps to retrieve more fine-grained details.
# 
# By default, the sentence window is 5 sentences on either side of the original sentence.
# 

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# ## Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index import ServiceContext, set_global_service_context
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
from llama_index.node_parser import (
    SentenceWindowNodeParser,
)
from llama_index.text_splitter import SentenceSplitter

# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# base node parser is a sentence splitter
text_splitter = SentenceSplitter()

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2", max_length=512
)
ctx = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    # node_parser=node_parser,
)

# if you wanted to use OpenAIEmbedding, we should also increase the batch size,
# since it involves many more calls to the API
# ctx = ServiceContext.from_defaults(llm=llm, embed_model=OpenAIEmbedding(embed_batch_size=50)), node_parser=node_parser)

# ## Load Data, Build the Index
# 

# ### Load Data
# 
# Here, we build an index using chapter 3 of the recent IPCC climate report.

#('curl https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter03.pdf --output IPCC_AR6_WGII_Chapter03.pdf')

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# ### Extract Nodes
# 
# We extract out the set of nodes that will be stored in the VectorIndex. This includes both the nodes with the sentence window parser, as well as the "base" nodes extracted using the standard parser.

nodes = node_parser.get_nodes_from_documents(documents)

base_nodes = text_splitter.get_nodes_from_documents(documents)

# ### Build the Indexes
# 
# We build both the sentence index, as well as the "base" index (with default chunk sizes).

from llama_index import VectorStoreIndex

sentence_index = VectorStoreIndex(nodes, service_context=ctx)

base_index = VectorStoreIndex(base_nodes, service_context=ctx)

# ## Querying
# 
# ### With MetadataReplacementPostProcessor
# 
# Here, we now use the `MetadataReplacementPostProcessor` to replace the sentence in each node with it's surrounding context.

from llama_index.postprocessor import MetadataReplacementPostProcessor

query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    # the target key defaults to `window` to match the node_parser's default
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
window_response = query_engine.query(
    "What are the concerns surrounding the AMOC?"
)
print(window_response)

# We can also check the original sentence that was retrieved for each node, as well as the actual window of sentences that was sent to the LLM.

window = window_response.source_nodes[0].node.metadata["window"]
sentence = window_response.source_nodes[0].node.metadata["original_text"]

print(f"Window: {window}")
print("------------------")
print(f"Original Sentence: {sentence}")

# ### Contrast with normal VectorStoreIndex

query_engine = base_index.as_query_engine(similarity_top_k=2)
vector_response = query_engine.query(
    "What are the concerns surrounding the AMOC?"
)
print(vector_response)

# Well, that didn't work. Let's bump up the top k! This will be slower and use more tokens compared to the sentence window index.

query_engine = base_index.as_query_engine(similarity_top_k=5)
vector_response = query_engine.query(
    "What are the concerns surrounding the AMOC?"
)
print(vector_response)

# ## Analysis
# 
# So the `SentenceWindowNodeParser` + `MetadataReplacementNodePostProcessor` combo is the clear winner here. But why?
# 
# Embeddings at a sentence level seem to capture more fine-grained details, like the word `AMOC`.
# 
# We can also compare the retrieved chunks for each index!

for source_node in window_response.source_nodes:
    print(source_node.node.metadata["original_text"])
    print("--------")

# Here, we can see that the sentence window index easily retrieved two nodes that talk about AMOC. Remember, the embeddings are based purely on the original sentence here, but the LLM actually ends up reading the surrounding context as well!

# Now, let's try and disect why the naive vector index failed.

for node in vector_response.source_nodes:
    print("AMOC mentioned?", "AMOC" in node.node.text)
    print("--------")

# So source node at index [2] mentions AMOC, but what did this text actually look like?

print(vector_response.source_nodes[2].node.text)

# So AMOC is disuccsed, but sadly it is in the middle chunk. With LLMs, it is often observed that text in the middle of retrieved context is often ignored or less useful. A recent paper ["Lost in the Middle" discusses this here](https://arxiv.org/abs/2307.03172).

# ## [Optional] Evaluation
# 
# We more rigorously evaluate how well the sentence window retriever works compared to the base retriever.
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
import random

nest_asyncio.apply()

len(base_nodes)

num_nodes_eval = 30
# there are 428 nodes total. Take the first 200 to generate questions (the back half of the doc is all references)
sample_eval_nodes = random.sample(base_nodes[:200], num_nodes_eval)
# NOTE: run this if the dataset isn't already saved
eval_service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4"))
# generate questions from the largest chunks (1024)
dataset_generator = DatasetGenerator(
    sample_eval_nodes,
    service_context=eval_service_context,
    show_progress=True,
    num_questions_per_chunk=2,
)

eval_dataset = await dataset_generator.agenerate_dataset_from_nodes()

eval_dataset.save_json("data/ipcc_eval_qr_dataset.json")

# optional
eval_dataset = QueryResponseDataset.from_json("data/ipcc_eval_qr_dataset.json")

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

max_samples = 30

eval_qs = eval_dataset.questions
ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]

# resetup base query engine and sentence window query engine
# base query engine
base_query_engine = base_index.as_query_engine(similarity_top_k=2)
# sentence window query engine
query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    # the target key defaults to `window` to match the node_parser's default
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)

import numpy as np

base_pred_responses = get_responses(
    eval_qs[:max_samples], base_query_engine, show_progress=True
)
pred_responses = get_responses(
    eval_qs[:max_samples], query_engine, show_progress=True
)

pred_response_strs = [str(p) for p in pred_responses]
base_pred_response_strs = [str(p) for p in base_pred_responses]

evaluator_dict = {
    "correctness": evaluator_c,
    "faithfulness": evaluator_f,
    "relevancy": evaluator_r,
    "semantic_similarity": evaluator_s,
}
batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)

# Run evaluations over faithfulness/semantic similarity.

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
    ["Sentence Window Retriever", "Base Retriever"],
    ["correctness", "relevancy", "faithfulness", "semantic_similarity"],
)
#display(results_df)

