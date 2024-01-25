#!/usr/bin/env python
# coding: utf-8

# # [WIP] Hyperparameter Optimization for RAG
# 
# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/param_optimizer/param_optimizer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 

# 
# We use our new, experimental `ParamTuner` class which allows hyperparameter grid search over a RAG function. It comes in two variants:
# 
# - `ParamTuner`: a naive way for parameter tuning by iterating over all parameters.
# - `RayTuneParamTuner`: a hyperparameter tuning mechanism powered by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)
# 
# The `ParamTuner` can take in any function that outputs a dictionary of values. In this setting we define a function that constructs a basic RAG ingestion pipeline from a set of documents (the Llama 2 paper), runs it over an evaluation dataset, and measures a correctness metric.
# 
# We investigate tuning the following parameters:
# 
# - Chunk size
# - Top k value

#('pip install llama-index')

#('mkdir data && wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"')

import nest_asyncio

nest_asyncio.apply()

from pathlib import Path
from llama_hub.file.pdf.base import PDFReader
from llama_hub.file.unstructured.base import UnstructuredReader
from llama_hub.file.pymu_pdf.base import PyMuPDFReader

loader = PDFReader()
docs0 = loader.load_data(file=Path("./data/llama2.pdf"))

from llama_index import Document

doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]

from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import IndexNode

# ## Load "Golden" Evaluation Dataset
# 
# Here we setup a "golden" evaluation dataset for the llama2 paper.
# 
# **NOTE**: We pull this in from Dropbox. For details on how to generate a dataset please see our `DatasetGenerator` module.

#('wget "https://www.dropbox.com/scl/fi/fh9vsmmm8vu0j50l3ss38/llama2_eval_qr_dataset.json?rlkey=kkoaez7aqeb4z25gzc06ak6kb&dl=1" -O data/llama2_eval_qr_dataset.json')

from llama_index.evaluation import QueryResponseDataset

# optional
eval_dataset = QueryResponseDataset.from_json(
    "data/llama2_eval_qr_dataset.json"
)

eval_qs = eval_dataset.questions
ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]

# ## Define Objective Function + Parameters
# 
# Here we define function to optimize given the parameters.
# 
# The function specifically does the following: 1) builds an index from documents, 2) queries index, and runs some basic evaluation.

from llama_index import (
    VectorStoreIndex,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
)
from llama_index.param_tuner.base import ParamTuner, TunedResult, RunResult
from llama_index.evaluation.eval_utils import get_responses, aget_responses
from llama_index.evaluation import SemanticSimilarityEvaluator, BatchEvalRunner
from llama_index.llms import OpenAI

import os
import numpy as np
from pathlib import Path

# ### Helper Functions

def _build_index(chunk_size, docs):
    index_out_path = f"./storage_{chunk_size}"
    if not os.path.exists(index_out_path):
        Path(index_out_path).mkdir(parents=True, exist_ok=True)
        # parse docs
        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
        base_nodes = node_parser.get_nodes_from_documents(docs)

        # build index
        index = VectorStoreIndex(base_nodes)
        # save index to disk
        index.storage_context.persist(index_out_path)
    else:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(
            persist_dir=index_out_path
        )
        # load index
        index = load_index_from_storage(
            storage_context,
        )
    return index

def _get_eval_batch_runner():
    eval_service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo")
    )
    evaluator_s = SemanticSimilarityEvaluator(
        service_context=eval_service_context
    )
    eval_batch_runner = BatchEvalRunner(
        {"semantic_similarity": evaluator_s}, workers=2, show_progress=True
    )

    return eval_batch_runner

# ### Objective Function (Sync)

def objective_function(params_dict):
    chunk_size = params_dict["chunk_size"]
    docs = params_dict["docs"]
    top_k = params_dict["top_k"]
    eval_qs = params_dict["eval_qs"]
    ref_response_strs = params_dict["ref_response_strs"]

    # build index
    index = _build_index(chunk_size, docs)

    # query engine
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    # get predicted responses
    pred_response_objs = get_responses(
        eval_qs, query_engine, show_progress=True
    )

    # run evaluator
    # NOTE: can uncomment other evaluators
    eval_batch_runner = _get_eval_batch_runner()
    eval_results = eval_batch_runner.evaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )

    # get semantic similarity metric
    mean_score = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    ).mean()

    return RunResult(score=mean_score, params=params_dict)

# ### Objective Function (Async)

async def aobjective_function(params_dict):
    chunk_size = params_dict["chunk_size"]
    docs = params_dict["docs"]
    top_k = params_dict["top_k"]
    eval_qs = params_dict["eval_qs"]
    ref_response_strs = params_dict["ref_response_strs"]

    # build index
    index = _build_index(chunk_size, docs)

    # query engine
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    # get predicted responses
    pred_response_objs = await aget_responses(
        eval_qs, query_engine, show_progress=True
    )

    # run evaluator
    # NOTE: can uncomment other evaluators
    eval_batch_runner = _get_eval_batch_runner()
    eval_results = await eval_batch_runner.aevaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )

    # get semantic similarity metric
    mean_score = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    ).mean()

    return RunResult(score=mean_score, params=params_dict)

# ### Parameters
# 
# We define both the parameters to grid-search over `param_dict` and fixed parameters `fixed_param_dict`.

param_dict = {"chunk_size": [256, 512, 1024], "top_k": [1, 2, 5]}
# param_dict = {
#     "chunk_size": [256],
#     "top_k": [1]
# }
fixed_param_dict = {
    "docs": docs,
    "eval_qs": eval_qs[:10],
    "ref_response_strs": ref_response_strs[:10],
}

# ## Run ParamTuner (default)
# 
# Here we run our default param tuner, which iterates through all hyperparameter combinations either synchronously or in async.

from llama_index.param_tuner import ParamTuner

param_tuner = ParamTuner(
    param_fn=objective_function,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    show_progress=True,
)

results = param_tuner.tune()

best_result = results.best_run_result
best_top_k = results.best_run_result.params["top_k"]
best_chunk_size = results.best_run_result.params["chunk_size"]
print(f"Score: {best_result.score}")
print(f"Top-k: {best_top_k}")
print(f"Chunk size: {best_chunk_size}")

# adjust test_idx for additional testing
test_idx = 6
p = results.run_results[test_idx].params
(results.run_results[test_idx].score, p["top_k"], p["chunk_size"])

# ### Run ParamTuner (Async)
# 
# Run the async version.

from llama_index.param_tuner import AsyncParamTuner

aparam_tuner = AsyncParamTuner(
    aparam_fn=aobjective_function,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    num_workers=2,
    show_progress=True,
)

results = await aparam_tuner.atune()

best_result = results.best_run_result
best_top_k = results.best_run_result.params["top_k"]
best_chunk_size = results.best_run_result.params["chunk_size"]
print(f"Score: {best_result.score}")
print(f"Top-k: {best_top_k}")
print(f"Chunk size: {best_chunk_size}")

# ## Run ParamTuner (Ray Tune)
# 
# Here we run our tuner powered by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html), a library for scalable hyperparameter tuning.
# 

from llama_index.param_tuner.base import RayTuneParamTuner

param_tuner = RayTuneParamTuner(
    param_fn=objective_function,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    run_config_dict={"storage_path": "/tmp/custom/ray_tune", "name": "my_exp"},
)

results = param_tuner.tune()

results.best_run_result.params.keys()

results.best_idx

best_result = results.best_run_result

best_top_k = results.best_run_result.params["top_k"]
best_chunk_size = results.best_run_result.params["chunk_size"]
print(f"Score: {best_result.score}")
print(f"Top-k: {best_top_k}")
print(f"Chunk size: {best_chunk_size}")

