#!/usr/bin/env python
# coding: utf-8

# # Knowledge Distillation For Fine-Tuning A GPT-3.5 Judge (Pairwise)
# 
# There has been recent research that demonstrated GPT-4's ability to closely align to human judges when evaluating LLM generated texts (e.g., see [[1]](https://arxiv.org/abs/2306.05685), [[2]](https://arxiv.org/abs/2303.16634)). In this notebook, we demonstrate how to use the `llama_index` library to distill knowledge from GPT-4 to GPT-3.5 so that a smaller GPT-3.5 becomes closer to GPT-4 performance; and by proxy, closer to human judges.
# 
# To do so, we will perform the following high level steps:
# 
# 1. Generate datasets: `train_dataset` and `test_dataset`
# 2. Perform knowledge distillation (using `train_dataset`)
# 3. Evaluate the distilled model  on `test_dataset`
# 
# 

# NOTE: this notebook makes several API calls to generate text with OpenAI GPT
# models as well as models hosted on HuggingFace. If you prefer not to wait for
# these generations, then the data for this notebook can be obtained with the
# `wget` command provided below.

# !wget "https://www.dropbox.com/scl/fo/m7skpjdbpb0g3p76y6epe/h?rlkey=omh2ysgh9qqqztf81qvjlivu2&dl=1" -O pairwise.zip

import nest_asyncio

nest_asyncio.apply()

import os

# we will be using models on HuggingFace as our LLM answer generators
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# we will use GPT-4 and GPT-3.5 + OpenAI Fine-Tuning
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import pandas as pd

# define jupyter #display function
def #display_eval_df(question, source, answer_a, answer_b, result) -> None:
    """Pretty print question/answer + gpt-4 judgement dataset."""
    eval_df = pd.DataFrame(
        {
            "Question": question,
            "Source": source,
            "Model A": answer_a["model"],
            "Answer A": answer_a["text"],
            "Model B": answer_b["model"],
            "Answer B": answer_b["text"],
            "Score": result.score,
            "Judgement": result.feedback,
        },
        index=[0],
    )
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "300px",
            "overflow-wrap": "break-word",
        },
        subset=["Answer A", "Answer B"]
    )
    #display(eval_df)

# ## Step 1 Generate datasets: `train_dataset` and `test_dataset`
# 
# For our dataset on which we will generate questions and prompt various LLMs to answer, we're going to use the `WikipediaReader` to read "History of <city>" for several cities. We're going to split up our cities into two lists: one to be used for `train_dataset` and the other for `test_dataset`.

#('pip install wikipedia -q')

# wikipedia pages
from llama_index.readers import WikipediaReader

train_cities = [
    "San Francisco",
    "Toronto",
    "New York",
    "Vancouver",
    "Montreal",
    "Boston",
]

test_cities = [
    "Tokyo",
    "Singapore",
    "Paris",
]

train_documents = WikipediaReader().load_data(
    pages=[f"History of {x}" for x in train_cities]
)
test_documents = WikipediaReader().load_data(
    pages=[f"History of {x}" for x in test_cities]
)

# ### Use a `DatasetGenerator` to build `train_dataset` and `test_dataset`
# 
# Now that we have our train and test set of `Document`'s, the next step is to generate the questions. For this we will use the `DatasetGenerator`, which uses an LLM to generate questions from given set of documents.

# #### Generate Questions

QUESTION_GEN_PROMPT = (
    "You are a Teacher/ Professor. Your task is to setup "
    "a quiz/examination. Using the provided context, formulate "
    "a single question that captures an important fact from the "
    "context. Restrict the question to the context information provided."
)

# With all that out of the way, let's spring into action. First, we will download the reference pdf document and create the set of questions against it.

# generate questions against chunks
from llama_index.evaluation import DatasetGenerator
from llama_index.llms import OpenAI
from llama_index import ServiceContext

# set context for llm provider
gpt_35_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3)
)

# instantiate DatasetGenerator's for train and test
train_dataset_generator = DatasetGenerator.from_documents(
    train_documents,
    question_gen_query=QUESTION_GEN_PROMPT,
    service_context=gpt_35_context,
    show_progress=True,
    num_questions_per_chunk=25,
)

test_dataset_generator = DatasetGenerator.from_documents(
    test_documents,
    question_gen_query=QUESTION_GEN_PROMPT,
    service_context=gpt_35_context,
    show_progress=True,
    num_questions_per_chunk=25,
)

# use DatasetGenerator to create questions from nodes
train_questions = train_dataset_generator.generate_questions_from_nodes(
    num=200
)

test_questions = test_dataset_generator.generate_questions_from_nodes(num=150)

len(train_questions), len(test_questions)

# let's take a look at a few of these
train_questions[:3]

test_questions[:3]

# #### Generate Answers To The Questions
# 
# The next step is to generate answers using LLMs. Just a reminder, that the point is to judge these generated answers. So later on, we will use GPT models to judge these answers.
# 
# But for the generation of the answers to the questions, we will use two other LLMs, namely: Llama-2 and Mistral. In order to do this, we first a create a vector store for our documents and an associated retriever, which both of the LLM answer-generators will use.

from llama_index import VectorStoreIndex
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever

# Create vector index
train_index = VectorStoreIndex.from_documents(documents=train_documents)

# Create the retriver on this index
train_retriever = VectorIndexRetriever(
    index=train_index,
    similarity_top_k=2,
)

# Create vector index for test to be used later
test_index = VectorStoreIndex.from_documents(documents=test_documents)

# Create the retriver for test to be used later
test_retriever = VectorIndexRetriever(
    index=test_index,
    similarity_top_k=2,
)

# From here we will build `RetrieverQueryEngine`'s that will take in our queries (i.e. questions) for processing. Note that we use `HuggingFaceInferenceAPI` for our LLM answer-generators, and that Llama-2 requires permissions. If you haven't yet gain accessed to these models, then feel free to swap out Llama-2 with another model of your choosing.

from llama_index.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index.llms import HuggingFaceInferenceAPI

def create_query_engine(
    hf_name: str, retriever: VectorIndexRetriever
) -> RetrieverQueryEngine:
    """Create a RetrieverQueryEngine using the HuggingFaceInferenceAPI LLM"""
    if hf_name not in hf_llm_generators:
        raise KeyError("model not listed in hf_llm_generators")
    llm = HuggingFaceInferenceAPI(
        model_name=hf_llm_generators[hf_name],
        context_window=2048,  # to use refine
        token=HUGGING_FACE_TOKEN,
    )
    context = ServiceContext.from_defaults(llm=llm)
    return RetrieverQueryEngine.from_args(
        retriever=retriever, service_context=context
    )

# define our llm-generators (query_engines)
hf_llm_generators = {
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
}

train_query_engines = {
    mdl: create_query_engine(mdl, train_retriever)
    for mdl in hf_llm_generators.keys()
}

test_query_engines = {
    mdl: create_query_engine(mdl, test_retriever)
    for mdl in hf_llm_generators.keys()
}

# We're ready to now to produce the answers from the various LLMs. We'll do this now for the `train_dataset` and hold off on doing this for `test_dataset` until the time comes for us to use it.
# 
# NOTE: this will take some time to generate. If you'd rather not wait, you have the option of loading the `train_qa.jsonl` that contains Llama-2 and Mistral answers per question.

import tqdm
import random

train_dataset = []
for q in tqdm.tqdm(train_questions):
    # randomly select two LLMs to generate answers to this q
    model_versus = random.sample(list(train_query_engines.items()), 2)

    # data for this q
    data_entry = {"question": q}
    responses = []
    source = None

    # generate answers
    for name, engine in model_versus:
        response = engine.query(q)
        response_struct = {}
        response_struct["model"] = name
        response_struct["text"] = str(response)
        if source is not None:
            assert source == response.source_nodes[0].node.text[:1000] + "..."
        else:
            source = response.source_nodes[0].node.text[:1000] + "..."
        responses.append(response_struct)

    data_entry["answers"] = responses
    data_entry["source"] = source
    train_dataset.append(data_entry)

# ### Get GPT-4 Evaluations On The Mistral and LLama-2 Answers 
# 
# As mentioned a couple of times before, the point of this guide is fine-tune an LLM judge from a GPT-4 judge. So, in order to complete our `train_dataset` we now need to instantiate our GPT-4 judge and have it evaluate the answers that were provided by the other LLMs: Llama-2 and Mistral. To do this, we will use the `PairwiseComparisonEvaluator` class. What this judge will do then is it will compare the two answers and provide a verdict as to whether Llama-2's answer is better, Mistral's answer is better, or if it's a tie.
# 
# There is a bit of added nuance here since with pairwise evaluations, we have to be mindful of the potential for "position-bias". This is when the judge favours the first answer that was presented to it (within the prompt/context). To account for this position-bias, we invoke the GPT-4 judge to perform to evaluations per sample, where in the second evaluation, we switch the order of presentation of the two answers (i.e., first evaluation: Llama-2 then Mistral, second evaluation: Mistral then Llama-2).
# 
# Finally, we also use the `OpenAIFineTuningHandler` which will collect all the chat histories that we will eventually need to fine-tune GPT-3.5.
# 
# NOTE: this will take some time to generate the judgements. Again, you have the option to load the `train_qa.jsonl` as `train_dataset`. Moreover, we also stored the JSONL files that we passed to OpenAI to fine-tune GPT-3.5.

# instantiate the gpt-4 judge
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index.callbacks import OpenAIFineTuningHandler
from llama_index.callbacks import CallbackManager
from llama_index.evaluation import PairwiseComparisonEvaluator

# NOTE: this finetuning_handler will collect 2x chat_histories for
# each query: one for original, and another for flipped
main_finetuning_handler = OpenAIFineTuningHandler()
callback_manager = CallbackManager([main_finetuning_handler])
gpt_4_context = ServiceContext.from_defaults(
    llm=OpenAI(temperature=0, model="gpt-4"),
    callback_manager=callback_manager,
)

gpt4_judge = PairwiseComparisonEvaluator(service_context=gpt_4_context)

for data_entry in tqdm.tqdm(train_dataset):
    final_eval_result = await gpt4_judge.aevaluate(
        query=data_entry["question"],
        response=data_entry["answers"][0]["text"],
        second_response=data_entry["answers"][1]["text"],
        reference=data_entry["source"],
    )

    # save final result
    judgement = {}
    judgement["llm"] = "gpt_4"
    judgement["score"] = final_eval_result.score
    judgement["text"] = final_eval_result.response
    judgement["source"] = final_eval_result.pairwise_source
    data_entry["evaluations"] = [judgement]

# Let's see how one of these GPT-4 evaluations looks like.

# let's see the last one
#display_eval_df(
    question=data_entry["question"],
    source=data_entry["source"],
    answer_a=data_entry["answers"][0],
    answer_b=data_entry["answers"][1],
    result=final_eval_result,
)

# #### Special Care To The Fine-Tuning JSONL
# 
# Since there are two evaluations (one for original order of presentation of the LLM answers and another for a flipped ordering), we need to be careful to choose the correct one to keep in our fine-tuning dataset. What this means is that we need to pick off the correct events that were collected by our `OpenAIFineTuningHandler` and then only use those to prepare the JSONL which we will pass to OpenAI's fine-tuning API.

main_finetuning_handler.save_finetuning_events(
    "pairwise_finetuning_events.jsonl"
)

# Get the fine_tuning_examples master dataset
with open("pairwise_finetuning_events.jsonl") as f:
    combined_finetuning_events = [json.loads(line) for line in f]

finetuning_events = (
    []
)  # for storing events using original order of presentation
flipped_finetuning_events = (
    []
)  # for storing events using flipped order of presentation

for ix, event in enumerate(combined_finetuning_events):
    if ix % 2 == 0:  # we always do original ordering first
        finetuning_events += [event]
    else:  # then we flip order and have GPT-4 make another judgement
        flipped_finetuning_events += [event]

assert len(finetuning_events) == len(flipped_finetuning_events)

# we need to pick which of the chat_histories to keep
resolved_finetuning_events = []
for ix, data_entry in enumerate(train_dataset):
    if data_entry["evaluations"][0]["source"] == "original":
        resolved_finetuning_events += [finetuning_events[ix]]
    elif data_entry["evaluations"][0]["source"] == "flipped":
        resolved_finetuning_events += [flipped_finetuning_events[ix]]
    else:
        continue

with open("resolved_pairwise_finetuning_events.jsonl", "w") as outfile:
    for entry in resolved_finetuning_events:
        print(json.dumps(entry), file=outfile)

# ## Step 2 Perform knowledge distillation
# 
# Okay, it's now time to distill some knowledge from GPT-4 to GPT-3.5 To do this, we will make use of the `OpenAIFinetuneEngine` class as well as the `resolved_pairwise_finetuning_events.jsonl` file that we just created.

from llama_index.finetuning import OpenAIFinetuneEngine

finetune_engine = OpenAIFinetuneEngine(
    "gpt-3.5-turbo",
    "resolved_pairwise_finetuning_events.jsonl",
)

finetune_engine.finetune()

# We can check the status of our current job as follows
# This may take some time ...
finetune_engine.get_current_job()

# ## 3 Evaluate The Fine-Tuned GPT-3.5 Judge On The Test Dataset
# 
# Now that we have our fine-tuned GPT-3.5, let's see how well it performs on a test set. But first, remember that we said we'd hold off on creating the `test_dataset` until the time comes that we need it? Well, that time is now. So we will repeat the process of creating the `train_dataset` here but instead now for the `test_dataset`.
# 
# NOTE: generating these answers and evaluations will take some time. You have the option of loading `test_qa_complete.jsonl` which has all the evaluations from the three considered LLM judges. You can load that as `test_dataset` and run the code found in the Metrics subsection below.

import random

# Use Llama-2 and Mistral LLMs to generate the answers to the test queries
test_dataset = []
for q in tqdm.tqdm(test_questions):
    # randomly select two LLMs to generate answers to this q
    model_versus = random.sample(list(test_query_engines.items()), 2)

    # data for this q
    data_entry = {"question": q}
    responses = []
    source = None

    # generate answers
    for name, engine in model_versus:
        response = engine.query(q)
        response_struct = {}
        response_struct["model"] = name
        response_struct["text"] = str(response)
        if source is not None:
            assert source == response.source_nodes[0].node.text[:1000] + "..."
        else:
            source = response.source_nodes[0].node.text[:1000] + "..."
        responses.append(response_struct)

    data_entry["answers"] = responses
    data_entry["source"] = source
    test_dataset.append(data_entry)

# get the gpt-4 judgments on the Mistal and Llama-2 answers
for data_entry in tqdm.tqdm(test_dataset):
    final_eval_result = await gpt4_judge.aevaluate(
        query=data_entry["question"],
        response=data_entry["answers"][0]["text"],
        second_response=data_entry["answers"][1]["text"],
        reference=data_entry["source"],
    )

    # save final result
    judgement = {}
    judgement["llm"] = "gpt_4"
    judgement["score"] = final_eval_result.score
    judgement["text"] = final_eval_result.response
    judgement["source"] = final_eval_result.pairwise_source
    data_entry["evaluations"] = [judgement]

from llama_index.evaluation import EvaluationResult

# use our fine-tuned GPT-3.5 to evaluate the answers
ft_llm = finetune_engine.get_finetuned_model()

ft_context = ServiceContext.from_defaults(
    llm=ft_llm,
)
ft_gpt_3p5_judge = PairwiseComparisonEvaluator(service_context=ft_context)

for data_entry in tqdm.tqdm(test_dataset):
    try:
        final_eval_result = await ft_gpt_3p5_judge.aevaluate(
            query=data_entry["question"],
            response=data_entry["answers"][0]["text"],
            second_response=data_entry["answers"][1]["text"],
            reference=data_entry["source"],
        )
    except:
        final_eval_result = EvaluationResult(
            query=eval_result.query,
            response="",
            passing=None,
            score=0.5,
            feedback="",
            pairwise_source="output-cannot-be-parsed",
        )

    # save final result
    judgement = {}
    judgement["llm"] = "ft_gpt_3p5"
    judgement["score"] = final_eval_result.score
    judgement["text"] = final_eval_result.response
    judgement["source"] = final_eval_result.pairwise_source
    data_entry["evaluations"] += [judgement]

# Similarly, use a non-fine-tuned judge to evaluate the answers
gpt_3p5_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo")
)
gpt_3p5_judge = PairwiseComparisonEvaluator(service_context=gpt_3p5_context)

for data_entry in tqdm.tqdm(test_dataset):
    try:
        final_eval_result = await gpt_3p5_judge.aevaluate(
            query=data_entry["question"],
            response=data_entry["answers"][0]["text"],
            second_response=data_entry["answers"][1]["text"],
            reference=data_entry["source"],
        )
    except:
        final_eval_result = EvaluationResult(
            query=data_entry["question"],
            response="",
            passing=None,
            score=0.5,
            feedback="",
            pairwise_source="output-cannot-be-parsed",
        )

    # save final result
    judgement = {}
    judgement["llm"] = "gpt_3p5"
    judgement["score"] = final_eval_result.score
    judgement["text"] = final_eval_result.response
    judgement["source"] = final_eval_result.pairwise_source
    data_entry["evaluations"] += [judgement]

# ### The Metrics
# 
# Phew! Now that we have generated all of the LLM judges evaluations of the Llama-2/Mistral answers on the test queries. Let's now get a quantitative view on how close fine-tuned GPT-3.5 is to GPT-4.
# 
# For this, we report several metrics, namely:
# - Agreement Rate with GPT-4 evaluations
# - Correlation to GPT-4 evaluations
# - Jaccard Similarity to GPT-4 evaluations
# 
# We also report the "inconclusive" counts, which is when the LLM judge switches its decision after being presented with the flipped order of presentation of Llama-2 and Mistral answers. Higher inconclusive counts is an indication of the LLM judge being susceptible to position bias, which is no good!

#('pip install scikit-learn -q')

import numpy as np

# store the scores and inconclusive booleans for each sample per LLM judge
scores = {"gpt_4": [], "gpt_3p5": [], "ft_gpt_3p5": []}
inconclusives = {"gpt_4": [], "gpt_3p5": [], "ft_gpt_3p5": []}

for ix, d in enumerate(test_dataset):
    for e in d["evaluations"]:
        scores[e["llm"]].append(e["score"])
        inconclusives[e["llm"]].append(
            e["source"] not in ["original", "flipped"]
        )

REPORT_FMT_STR = (
    "{model}\n"
    "-----------------\n"
    "Number of inconclusives: {inconclusive}\n"
    "Number of agreements with GPT-4: {agreement} out of {total}\n"
    "Agreement rate: {agreement_rate}\n"
    "Correlation: {corr}\n"
    "Jaccard: {jacc}\n\n"
)

from sklearn.metrics import jaccard_score

# numpy conversion
np_scores_gpt_4 = np.array(scores["gpt_4"])
np_scores_gpt_3p5 = np.array(scores["gpt_3p5"])
np_scores_ft_gpt_3p5 = np.array(scores["ft_gpt_3p5"])

# can only compare when both judges have non inconclusive results
ft_mask = ~np.array(inconclusives["gpt_4"]) * ~np.array(
    inconclusives["ft_gpt_3p5"]
)
no_ft_mask = ~np.array(inconclusives["gpt_4"]) * ~np.array(
    inconclusives["gpt_3p5"]
)

# agreement rates
agreement_ft = sum(np_scores_gpt_4[ft_mask] == np_scores_ft_gpt_3p5[ft_mask])
agreement_rate_ft = agreement_ft / sum(ft_mask)
agreement_no_ft = sum(
    np_scores_gpt_4[no_ft_mask] == np_scores_gpt_3p5[no_ft_mask]
)
agreement_rate_no_ft = agreement_no_ft / sum(no_ft_mask)

# correlations
corr_ft = np.corrcoef(np_scores_gpt_4[ft_mask], np_scores_ft_gpt_3p5[ft_mask])[
    0, 1
]
corr_no_ft = np.corrcoef(
    np_scores_gpt_4[no_ft_mask], np_scores_gpt_3p5[no_ft_mask]
)[0, 1]

# jaccard
jaccard_ft = jaccard_score(
    np_scores_gpt_4[ft_mask].astype(str),
    np_scores_ft_gpt_3p5[ft_mask].astype(str),
    average="weighted",
)
jaccard_no_ft = jaccard_score(
    np_scores_gpt_4[no_ft_mask].astype(str),
    np_scores_gpt_3p5[no_ft_mask].astype(str),
    average="weighted",
)

print(
    REPORT_FMT_STR.format(
        model="GPT-3.5 w/ fine-tuning",
        inconclusive=sum(inconclusives["ft_gpt_3p5"]),
        agreement=agreement_ft,
        total=sum(ft_mask),
        agreement_rate=agreement_rate_ft,
        corr=corr_ft,
        jacc=jaccard_ft,
    )
)
print(
    REPORT_FMT_STR.format(
        model="GPT-3.5 w/out fine-tuning",
        inconclusive=sum(inconclusives["gpt_3p5"]),
        agreement=agreement_no_ft,
        total=sum(no_ft_mask),
        agreement_rate=agreement_rate_no_ft,
        corr=corr_no_ft,
        jacc=jaccard_no_ft,
    )
)
print(
    f"GPT-4\n-----------------\nInconclusive Count: {sum(inconclusives['gpt_4'])}"
)

# ## Conclusion
# 
# From the above numbers we see that fine-tuning a GPT-3.5 judge yields higher agreement scores, correlation, and jaccard similarity than a non-fine-tuned GPT-3.5 judge. What's more is that we see the inconclusive counts go down after fine-tuning as well. Overall, we see that fine-tuning here has helped us to get a GPT-3.5 judge that is closer to a GPT-4 judge (and thus by proxy, closer to human judgements) and at the same time helped remedy the position bias that a non-fine-tuned GPT-3.5 would have otherwise.
