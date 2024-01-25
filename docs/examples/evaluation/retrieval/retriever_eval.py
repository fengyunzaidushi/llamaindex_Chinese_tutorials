#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/evaluation/retrieval/retriever_eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Retrieval Evaluation
# 
# This notebook uses our `RetrieverEvaluator` to evaluate the quality of any Retriever module defined in LlamaIndex.
# 
# We specify a set of different evaluation metrics: this includes hit-rate and MRR. For any given question, these will compare the quality of retrieved results from the ground-truth context.
# 
# To ease the burden of creating the eval dataset in the first place, we can rely on synthetic data generation.

# ## Setup
# 
# Here we load in data (PG essay), parse into Nodes. We then index this data using our simple vector index and get a retriever.

import nest_asyncio

nest_asyncio.apply()

from llama_index.evaluation import generate_question_context_pairs
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SentenceSplitter
from llama_index.llms import OpenAI

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

node_parser = SentenceSplitter(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)

# by default, the node ids are set to random uuids. To ensure same id's per run, we manually set them.
for idx, node in enumerate(nodes):
    node.id_ = f"node_{idx}"

llm = OpenAI(model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)

vector_index = VectorStoreIndex(nodes, service_context=service_context)
retriever = vector_index.as_retriever(similarity_top_k=2)

# ### Try out Retrieval
# 
# We'll try out retrieval over a simple dataset.

retrieved_nodes = retriever.retrieve("What did the author do growing up?")

from llama_index.response.notebook_utils import #display_source_node

for node in retrieved_nodes:
    #display_source_node(node, source_length=1000)

# ## Build an Evaluation dataset of (query, context) pairs
# 
# Here we build a simple evaluation dataset over the existing text corpus.
# 
# We use our `generate_question_context_pairs` to generate a set of (question, context) pairs over a given unstructured text corpus. This uses the LLM to auto-generate questions from each context chunk.
# 
# We get back a `EmbeddingQAFinetuneDataset` object. At a high-level this contains a set of ids mapping to queries and relevant doc chunks, as well as the corpus itself.

from llama_index.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)

qa_dataset = generate_question_context_pairs(
    nodes, llm=llm, num_questions_per_chunk=2
)

queries = qa_dataset.queries.values()
print(list(queries)[2])

# [optional] save
qa_dataset.save_json("pg_eval_dataset.json")

# [optional] load
qa_dataset = EmbeddingQAFinetuneDataset.from_json("pg_eval_dataset.json")

# ## Use `RetrieverEvaluator` for Retrieval Evaluation
# 
# We're now ready to run our retrieval evals. We'll run our `RetrieverEvaluator` over the eval dataset that we generated.

# We define two functions: `get_eval_results` and also `#display_results` that run our retriever over the dataset.

include_cohere_rerank = True

if include_cohere_rerank:
    #('pip install cohere -q')

from llama_index.evaluation import RetrieverEvaluator

metrics = ["mrr", "hit_rate"]

if include_cohere_rerank:
    metrics.append(
        "cohere_rerank_relevancy"  # requires COHERE_API_KEY environment variable to be set
    )

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    metrics, retriever=retriever
)

# try it out on a sample query
sample_id, sample_query = list(qa_dataset.queries.items())[0]
sample_expected = qa_dataset.relevant_docs[sample_id]

eval_result = retriever_evaluator.evaluate(sample_query, sample_expected)
print(eval_result)

# try it out on an entire dataset
eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)

import pandas as pd

def #display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    columns = {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}

    if include_cohere_rerank:
        crr_relevancy = full_df["cohere_rerank_relevancy"].mean()
        columns.update({"cohere_rerank_relevancy": [crr_relevancy]})

    metric_df = pd.DataFrame(columns)

    return metric_df

#display_results("top-2 eval", eval_results)

