#!/usr/bin/env python
# coding: utf-8

# # Benchmarking OpenAI Retrieval API (through Assistant Agent)
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/agent/openai_retrieval_benchmark.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# This guide benchmarks the Retrieval Tool from the [OpenAI Assistant API](https://platform.openai.com/docs/assistants/overview), by using our `OpenAIAssistantAgent`. We run over the Llama 2 paper, and compare generation quality against a naive RAG pipeline.
# 

#('pip install llama-index')

import nest_asyncio

nest_asyncio.apply()

# ## Setup Data
# 
# Here we load the Llama 2 paper and chunk it.

#("mkdir -p 'data/'")
#('wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"')

from pathlib import Path
from llama_index import Document, ServiceContext, VectorStoreIndex
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI

loader = PyMuPDFReader()
docs0 = loader.load(file_path=Path("./data/llama2.pdf"))

doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]

node_parser = SimpleNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(docs)

len(nodes)

# ## Define Eval Modules
# 
# We setup evaluation modules, including the dataset and evaluators.

# ### Setup "Golden Dataset"
# 
# Here we load in a "golden" dataset.

# #### Option 1: Pull Existing Dataset
# 
# **NOTE**: We pull this in from Dropbox. For details on how to generate a dataset please see our `DatasetGenerator` module.

#('wget "https://www.dropbox.com/scl/fi/fh9vsmmm8vu0j50l3ss38/llama2_eval_qr_dataset.json?rlkey=kkoaez7aqeb4z25gzc06ak6kb&dl=1" -O data/llama2_eval_qr_dataset.json')

from llama_index.evaluation import QueryResponseDataset

# optional
eval_dataset = QueryResponseDataset.from_json(
    "data/llama2_eval_qr_dataset.json"
)

# #### Option 2: Generate New Dataset
# 
# If you choose this option, you can choose to generate a new dataset from scratch. This allows you to play around with our `DatasetGenerator` settings to make sure it suits your needs.

from llama_index.evaluation import (
    DatasetGenerator,
    QueryResponseDataset,
)
from llama_index import ServiceContext
from llama_index.llms import OpenAI

# NOTE: run this if the dataset isn't already saved
# Note: we only generate from the first 20 nodes, since the rest are references
eval_service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4-1106-preview")
)
dataset_generator = DatasetGenerator(
    nodes[:20],
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

# ### Eval Modules
# 
# We define two evaluation modules: correctness and semantic similarity - both comparing quality of predicted response with actual response.

from llama_index.evaluation.eval_utils import get_responses, get_results_df
from llama_index.evaluation import (
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    BatchEvalRunner,
)
from llama_index.llms import OpenAI

eval_llm = OpenAI(model="gpt-4-1106-preview")
eval_service_context = ServiceContext.from_defaults(llm=eval_llm)
evaluator_c = CorrectnessEvaluator(service_context=eval_service_context)
evaluator_s = SemanticSimilarityEvaluator(service_context=eval_service_context)
evaluator_dict = {
    "correctness": evaluator_c,
    "semantic_similarity": evaluator_s,
}
batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)

import numpy as np
import time
import os
import pickle
from tqdm import tqdm

def get_responses_sync(
    eval_qs, query_engine, show_progress=True, save_path=None
):
    if show_progress:
        eval_qs_iter = tqdm(eval_qs)
    else:
        eval_qs_iter = eval_qs
    pred_responses = []
    start_time = time.time()
    for eval_q in eval_qs_iter:
        print(f"eval q: {eval_q}")
        pred_response = agent.query(eval_q)
        print(f"predicted response: {pred_response}")
        pred_responses.append(pred_response)
        if save_path is not None:
            # save intermediate responses (to cache in case something breaks)
            avg_time = (time.time() - start_time) / len(pred_responses)
            pickle.dump(
                {"pred_responses": pred_responses, "avg_time": avg_time},
                open(save_path, "wb"),
            )
    return pred_responses

async def run_evals(
    query_engine,
    eval_qa_pairs,
    batch_runner,
    disable_async_for_preds=False,
    save_path=None,
):
    # then evaluate
    # TODO: evaluate a sample of generated results
    eval_qs = [q for q, _ in eval_qa_pairs]
    eval_answers = [a for _, a in eval_qa_pairs]

    if save_path is not None:
        if not os.path.exists(save_path):
            start_time = time.time()
            if disable_async_for_preds:
                pred_responses = get_responses_sync(
                    eval_qs,
                    query_engine,
                    show_progress=True,
                    save_path=save_path,
                )
            else:
                pred_responses = get_responses(
                    eval_qs, query_engine, show_progress=True
                )
            avg_time = (time.time() - start_time) / len(eval_qs)
            pickle.dump(
                {"pred_responses": pred_responses, "avg_time": avg_time},
                open(save_path, "wb"),
            )
        else:
            # [optional] load
            pickled_dict = pickle.load(open(save_path, "rb"))
            pred_responses = pickled_dict["pred_responses"]
            avg_time = pickled_dict["avg_time"]
    else:
        start_time = time.time()
        pred_responses = get_responses(
            eval_qs, query_engine, show_progress=True
        )
        avg_time = (time.time() - start_time) / len(eval_qs)

    eval_results = await batch_runner.aevaluate_responses(
        eval_qs, responses=pred_responses, reference=eval_answers
    )
    return eval_results, {"avg_time": avg_time}

# ## Construct Assistant with Built-In Retrieval
# 
# Let's construct the assistant by also passing it the built-in OpenAI Retrieval tool.
# 
# Here, we upload and pass in the file during assistant-creation time. 

from llama_index.agent import OpenAIAssistantAgent

agent = OpenAIAssistantAgent.from_new(
    name="SEC Analyst",
    instructions="You are a QA assistant designed to analyze sec filings.",
    openai_tools=[{"type": "retrieval"}],
    instructions_prefix="Please address the user as Jerry.",
    files=["data/llama2.pdf"],
    verbose=True,
)

response = agent.query(
    "What are the key differences between Llama 2 and Llama 2-Chat?"
)

print(str(response))

# ## Benchmark
# 
# We run the agent over our evaluation dataset. We benchmark against a standard top-k RAG pipeline (k=2) with gpt-4-turbo.
# 
# **NOTE**: During our time of testing (November 2023), the Assistant API is heavily rate-limited, and can take ~1-2 hours to generate responses over 60 datapoints.

# #### Define Baseline Index + RAG Pipeline

base_sc = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-1106-preview"))
base_index = VectorStoreIndex(nodes, service_context=base_sc)
base_query_engine = base_index.as_query_engine(similarity_top_k=2)

# #### Run Evals over Baseline

base_eval_results, base_extra_info = await run_evals(
    base_query_engine,
    eval_dataset.qr_pairs,
    batch_runner,
    save_path="data/llama2_preds_base.pkl",
)

results_df = get_results_df(
    [base_eval_results],
    ["Base Query Engine"],
    ["correctness", "semantic_similarity"],
)
#display(results_df)

# #### Run Evals over Assistant API

assistant_eval_results, assistant_extra_info = await run_evals(
    agent,
    eval_dataset.qr_pairs[:55],
    batch_runner,
    save_path="data/llama2_preds_assistant.pkl",
    disable_async_for_preds=True,
)

# #### Get Results
# 
# Here we see...that our basic RAG pipeline does better.
# 
# Take these numbers with a grain of salt. The goal here is to give you a script so you can run this on your own data.
# 
# That said it's surprising the Retrieval API doesn't give immediately better out of the box performance.

results_df = get_results_df(
    [assistant_eval_results, base_eval_results],
    ["Retrieval API", "Base Query Engine"],
    ["correctness", "semantic_similarity"],
)
#display(results_df)
print(f"Base Avg Time: {base_extra_info['avg_time']}")
print(f"Assistant Avg Time: {assistant_extra_info['avg_time']}")

