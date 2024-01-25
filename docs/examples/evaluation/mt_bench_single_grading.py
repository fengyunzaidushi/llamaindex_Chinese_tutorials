#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/evaluation/mt_bench_single_grading.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Benchmarking LLM Evaluators On A Mini MT-Bench (Single Grading) `LabelledEvaluatorDataset`

# 
# 1. GPT-3.5 (OpenAI)
# 2. GPT-4 (OpenAI)
# 3. Gemini-Pro (Google)

import nest_asyncio

nest_asyncio.apply()

#('pip install "google-generativeai" -q')

# ### Load in Evaluator Dataset

# Let's load in the llama-dataset from llama-hub.

from llama_index.llama_dataset import download_llama_dataset

# download dataset
evaluator_dataset, _ = download_llama_dataset(
    "MiniMtBenchSingleGradingDataset", "./mini_mt_bench_data"
)

evaluator_dataset.to_pandas()[:5]

# ### Define Our Evaluators

from llama_index.evaluation import CorrectnessEvaluator
from llama_index.llms import OpenAI, Gemini, Cohere
from llama_index import ServiceContext

gpt_4_context = ServiceContext.from_defaults(
    llm=OpenAI(temperature=0, model="gpt-4"),
)

gpt_3p5_context = ServiceContext.from_defaults(
    llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
)

gemini_pro_context = ServiceContext.from_defaults(
    llm=Gemini(model="models/gemini-pro", temperature=0)
)

evaluators = {
    "gpt-4": CorrectnessEvaluator(service_context=gpt_4_context),
    "gpt-3.5": CorrectnessEvaluator(service_context=gpt_3p5_context),
    "gemini-pro": CorrectnessEvaluator(service_context=gemini_pro_context),
}

# ### Benchmark With `EvaluatorBenchmarkerPack` (llama-pack)
# 
# When using the `EvaluatorBenchmarkerPack` with a `LabelledEvaluatorDataset`, the returned benchmarks will contain values for the following quantites:
# 
# - `number_examples`: The number of examples the dataset consists of.
# - `invalid_predictions`: The number of evaluations that could not yield a final evaluation (e.g., due to inability to parse the evaluation output, or an exception thrown by the LLM evaluator)
# - `correlation`: The correlation between the scores of the provided evaluator and those of the reference evaluator (in this case gpt-4).
# - `mae`: The mean absolute error between the scores of the provided evaluator and those of the reference evaluator.
# - `hamming`: The hamming distance between the scores of the provided evaluator and those of the reference evaluator.
# 
# NOTE: `correlation`, `mae`, and `hamming` are all computed without invalid predictions. So, essentially these metrics are conditional ones, conditioned on the prediction being valid.

from llama_index.llama_pack import download_llama_pack

EvaluatorBenchmarkerPack = download_llama_pack(
    "EvaluatorBenchmarkerPack", "./pack"
)

# #### GPT 3.5

evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gpt-3.5"],
    eval_dataset=evaluator_dataset,
    show_progress=True,
)

gpt_3p5_benchmark_df = await evaluator_benchmarker.arun(
    batch_size=100, sleep_time_in_seconds=0
)

gpt_3p5_benchmark_df.index = ["gpt-3.5"]
gpt_3p5_benchmark_df

# #### GPT-4

evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gpt-4"],
    eval_dataset=evaluator_dataset,
    show_progress=True,
)

gpt_4_benchmark_df = await evaluator_benchmarker.arun(
    batch_size=100, sleep_time_in_seconds=0
)

gpt_4_benchmark_df.index = ["gpt-4"]
gpt_4_benchmark_df

# #### Gemini Pro

evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gemini-pro"],
    eval_dataset=evaluator_dataset,
    show_progress=True,
)

gemini_pro_benchmark_df = await evaluator_benchmarker.arun(
    batch_size=5, sleep_time_in_seconds=0.5
)

gemini_pro_benchmark_df.index = ["gemini-pro"]
gemini_pro_benchmark_df

evaluator_benchmarker.prediction_dataset.save_json(
    "mt_sg_gemini_predictions.json"
)

# ##
# 
# Putting all baselines together.

import pandas as pd

final_benchmark = pd.concat(
    [
        gpt_3p5_benchmark_df,
        gpt_4_benchmark_df,
        gemini_pro_benchmark_df,
    ],
    axis=0,
)
final_benchmark

# From the results above, we make the following observations:
# - GPT-3.5 and Gemini-Pro seem to have similar results, with perhaps the slightes edge to GPT-3.5 in terms of closeness to GPT-4.
# - Though, both don't seem to be too close to GPT-4.
# - GPT-4 seems to be pretty consistent with itself in this benchmark.
