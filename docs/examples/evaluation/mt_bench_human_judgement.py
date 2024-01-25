#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/evaluation/mt_bench_human_judgement.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Benchmarking LLM Evaluators On The MT-Bench Human Judgement `LabelledPairwiseEvaluatorDataset`

# 
# 1. GPT-3.5 (OpenAI)
# 2. GPT-4 (OpenAI)
# 3. Gemini-Pro (Google)

#('pip install "google-generativeai" -q')

import nest_asyncio

nest_asyncio.apply()

# ### Load In Dataset
# 
# Let's load in the llama-dataset from llama-hub.

from llama_index.llama_dataset import download_llama_dataset

# download dataset
pairwise_evaluator_dataset, _ = download_llama_dataset(
    "MtBenchHumanJudgementDataset", "./mt_bench_data"
)

pairwise_evaluator_dataset.to_pandas()[:5]

# ### Define Our Evaluators

from llama_index.evaluation import PairwiseComparisonEvaluator
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
    "gpt-4": PairwiseComparisonEvaluator(service_context=gpt_4_context),
    "gpt-3.5": PairwiseComparisonEvaluator(service_context=gpt_3p5_context),
    "gemini-pro": PairwiseComparisonEvaluator(
        service_context=gemini_pro_context
    ),
}

# ### Benchmark With `EvaluatorBenchmarkerPack` (llama-pack)
# 
# To compare our four evaluators we will benchmark them against `MTBenchHumanJudgementDataset`, wherein references are provided by human evaluators. The benchmarks will return the following quantites:
# 
# - `number_examples`: The number of examples the dataset consists of.
# - `invalid_predictions`: The number of evaluations that could not yield a final evaluation (e.g., due to inability to parse the evaluation output, or an exception thrown by the LLM evaluator)
# - `inconclusives`: Since this is a pairwise comparison, to mitigate the risk for "position bias" we conduct two evaluations — one with original order of presenting the two model answers, and another with the order in which these answers are presented to the evaluator LLM is flipped. A result is inconclusive if the LLM evaluator in the second ordering flips its vote in relation to the first vote.
# - `ties`: A `PairwiseComparisonEvaluator` can also return a "tie" result. This is the number of examples for which it gave a tie result.
# - `agreement_rate_with_ties`: The rate at which the LLM evaluator agrees with the reference (in this case human) evaluator, when also including ties. The denominator used to compute this metric is given by: `number_examples - invalid_predictions - inconclusives`.
# - `agreement_rate_without_ties`: The rate at which the LLM evaluator agress with the reference (in this case human) evaluator, when excluding and ties. The denominator used to compute this metric is given by: `number_examples - invalid_predictions - inconclusives - ties`.
# 
# To compute these metrics, we'll make use of the `EvaluatorBenchmarkerPack`.

from llama_index.llama_pack import download_llama_pack

EvaluatorBenchmarkerPack = download_llama_pack(
    "EvaluatorBenchmarkerPack", "./pack"
)

# #### GPT-3.5

evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gpt-3.5"],
    eval_dataset=pairwise_evaluator_dataset,
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
    eval_dataset=pairwise_evaluator_dataset,
    show_progress=True,
)

gpt_4_benchmark_df = await evaluator_benchmarker.arun(
    batch_size=100, sleep_time_in_seconds=0
)

gpt_4_benchmark_df.index = ["gpt-4"]
gpt_4_benchmark_df

# ### Gemini Pro
# 
# NOTE: The rate limit for Gemini models is still very constraining, which is understandable given that they've just been released at the time of writing this notebook. So, we use a very small `batch_size` and moderately high `sleep_time_in_seconds` to reduce risk of getting rate-limited.

evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gemini-pro"],
    eval_dataset=pairwise_evaluator_dataset,
    show_progress=True,
)

gemini_pro_benchmark_df = await evaluator_benchmarker.arun(
    batch_size=5, sleep_time_in_seconds=0.5
)

gemini_pro_benchmark_df.index = ["gemini-pro"]
gemini_pro_benchmark_df

evaluator_benchmarker.prediction_dataset.save_json("gemini_predictions.json")

# ### Summary
# 
# For convenience, let's put all the results in a single DataFrame.

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
# - In terms of agreement rates, all three models seem quite close, with perhaps a slight edge given to the Gemini models
# - Gemini Pro and GPT-3.5 seem to be a bit more assertive than GPT-4 resulting in only 50-60 ties to GPT-4's 100 ties.
# - However, perhaps related to the previous point, GPT-4 yields the least amount of inconclusives, meaning that it suffers the least from position bias.
# - Overall, it seems that Gemini Pro is up to snuff with GPT models, and would say that it outperforms GPT-3.5 — looks like Gemini can be legit alternatives to GPT models for evaluation tasks.
