#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/llama_datasets/downloading_llama_datasets.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Downloading a LlamaDataset from LlamaHub
# 
# You can browse our available benchmark datasets via [llamahub.ai](https://llamahub.ai/). This notebook guide depicts how you can download the dataset and its source text documents. In particular, the `download_llama_dataset` will download the evaluation dataset (i.e., `LabelledRagDataset`) as well as the `Document`'s of the source text files used to build the evaluation dataset in the first place.
# 
# Finally, in this notebook, we also demonstrate the end to end workflow of downloading an evaluation dataset, making predictions on it using your own RAG pipeline (query engine) and then evaluating these predictions.

from llama_index.llama_dataset import download_llama_dataset

# download and install dependencies
rag_dataset, documents = download_llama_dataset(
    "PaulGrahamEssayDataset", "./paul_graham"
)

rag_dataset.to_pandas()[:5]

# With `documents`, you can build your own RAG pipeline, to then predict and perform evaluations to compare against the benchmarks listed in the `DatasetCard` associated with the datasets [llamahub.ai](https://llamahub.ai/).

# ### Predictions
# 
# **NOTE**: The rest of the notebook illustrates how to manually perform predictions and subsequent evaluations for demonstrative purposes. Alternatively you can use the `RagEvaluatorPack` that will take care of predicting and evaluating using a RAG system that you would have provided.

from llama_index import VectorStoreIndex

# a basic RAG pipeline, uses service context defaults
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()

# You can now create predictions and perform evaluation manually or download the `PredictAndEvaluatePack` to do this for you in a single line of code.

import nest_asyncio

nest_asyncio.apply()

# manually
prediction_dataset = await rag_dataset.amake_predictions_with(
    query_engine=query_engine, show_progress=True
)

prediction_dataset.to_pandas()[:5]

# ### Evaluation
# 
# Now that we have our predictions, we can perform evaluations on two dimensions:
# 
# 1. The generated response: how well the predicted response matches the reference answer.
# 2. The retrieved contexts: how well the retrieved contexts for the prediction match the reference contexts.
# 
# NOTE: For retrieved contexts, we are unable to use standard retrieval metrics such as `hit rate` and `mean reciproccal rank` due to the fact that doing so requires we have the same index that was used to generate the ground truth data. But, it is not necessary for a `LabelledRagDataset` to be even created by an index. As such, we will use `semantic similarity` between the prediction's contexts and the reference contexts as a measure of goodness.

import tqdm

# For evaluating the response, we will use the LLM-As-A-Judge pattern. Specifically, we will use `CorrectnessEvaluator`, `FaithfulnessEvaluator` and `RelevancyEvaluator`.
# 
# For evaluating the goodness of the retrieved contexts we will use `SemanticSimilarityEvaluator`.

# instantiate the gpt-4 judge
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index.evaluation import (
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    SemanticSimilarityEvaluator,
)

judges = {}

judges["correctness"] = CorrectnessEvaluator(
    service_context=ServiceContext.from_defaults(
        llm=OpenAI(temperature=0, model="gpt-4"),
    )
)

judges["relevancy"] = RelevancyEvaluator(
    service_context=ServiceContext.from_defaults(
        llm=OpenAI(temperature=0, model="gpt-4"),
    )
)

judges["faithfulness"] = FaithfulnessEvaluator(
    service_context=ServiceContext.from_defaults(
        llm=OpenAI(temperature=0, model="gpt-4"),
    )
)

judges["semantic_similarity"] = SemanticSimilarityEvaluator(
    service_context=ServiceContext.from_defaults()
)

# Loop through the (`labelled_example`, `prediction`) pais and perform the evaluations on each of them individually.

evals = {
    "correctness": [],
    "relevancy": [],
    "faithfulness": [],
    "context_similarity": [],
}

for example, prediction in tqdm.tqdm(
    zip(rag_dataset.examples, prediction_dataset.predictions)
):
    correctness_result = judges["correctness"].evaluate(
        query=example.query,
        response=prediction.response,
        reference=example.reference_answer,
    )

    relevancy_result = judges["relevancy"].evaluate(
        query=example.query,
        response=prediction.response,
        contexts=prediction.contexts,
    )

    faithfulness_result = judges["faithfulness"].evaluate(
        query=example.query,
        response=prediction.response,
        contexts=prediction.contexts,
    )

    semantic_similarity_result = judges["semantic_similarity"].evaluate(
        query=example.query,
        response="\n".join(prediction.contexts),
        reference="\n".join(example.reference_contexts),
    )

    evals["correctness"].append(correctness_result)
    evals["relevancy"].append(relevancy_result)
    evals["faithfulness"].append(faithfulness_result)
    evals["context_similarity"].append(semantic_similarity_result)

import json

# saving evaluations
evaluations_objects = {
    "context_similarity": [e.dict() for e in evals["context_similarity"]],
    "correctness": [e.dict() for e in evals["correctness"]],
    "faithfulness": [e.dict() for e in evals["faithfulness"]],
    "relevancy": [e.dict() for e in evals["relevancy"]],
}

with open("evaluations.json", "w") as json_file:
    json.dump(evaluations_objects, json_file)

# Now, we can use our notebook utility functions to view these evaluations.

import pandas as pd
from llama_index.evaluation.notebook_utils import (
    get_eval_results_df,
)

deep_eval_df, mean_correctness_df = get_eval_results_df(
    ["base_rag"] * len(evals["correctness"]),
    evals["correctness"],
    metric="correctness",
)
deep_eval_df, mean_relevancy_df = get_eval_results_df(
    ["base_rag"] * len(evals["relevancy"]),
    evals["relevancy"],
    metric="relevancy",
)
_, mean_faithfulness_df = get_eval_results_df(
    ["base_rag"] * len(evals["faithfulness"]),
    evals["faithfulness"],
    metric="faithfulness",
)
_, mean_context_similarity_df = get_eval_results_df(
    ["base_rag"] * len(evals["context_similarity"]),
    evals["context_similarity"],
    metric="context_similarity",
)

mean_scores_df = pd.concat(
    [
        mean_correctness_df.reset_index(),
        mean_relevancy_df.reset_index(),
        mean_faithfulness_df.reset_index(),
        mean_context_similarity_df.reset_index(),
    ],
    axis=0,
    ignore_index=True,
)
mean_scores_df = mean_scores_df.set_index("index")
mean_scores_df.index = mean_scores_df.index.set_names(["metrics"])

mean_scores_df

# On this toy example, we see that the basic RAG pipeline performs quite well against the evaluation benchmark (`rag_dataset`)! For completeness, to perform the above steps instead by using the `RagEvaluatorPack`, use the code provided below:

from llama_index.llama_pack import download_llama_pack

RagEvaluatorPack = download_llama_pack("RagEvaluatorPack", "./pack")
rag_evaluator = RagEvaluatorPack(
    query_engine=query_engine, rag_dataset=rag_dataset, show_progress=True
)

############################################################################
# NOTE: If have a lower tier subscription for OpenAI API like Usage Tier 1 #
# then you'll need to use different batch_size and sleep_time_in_seconds.  #
# For Usage Tier 1, settings that seemed to work well were batch_size=5,   #
# and sleep_time_in_seconds=15 (as of December 2023.)                      #
############################################################################

benchmark_df = await rag_evaluator_pack.arun(
    batch_size=20,  # batches the number of openai api calls to make
    sleep_time_in_seconds=1,  # seconds to sleep before making an api call
)

