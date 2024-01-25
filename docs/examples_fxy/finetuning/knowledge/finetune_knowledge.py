#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/finetuning/knowledge/finetune_knowledge.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 

# # Fine-tuning to Memorize Knowledge
# 

# 
# - Synthesizing questions from existing context
# - Trying text completion

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import openai
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

# ## Load Data

#('mkdir data && wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"')

from pathlib import Path
from llama_hub.file.pdf.base import PDFReader
from llama_hub.file.unstructured.base import UnstructuredReader
from llama_hub.file.pymu_pdf.base import PyMuPDFReader

loader = PyMuPDFReader()
docs0 = loader.load(file_path=Path("./data/llama2.pdf"))

from llama_index import Document

doc_text = "\n\n".join([d.get_content() for d in docs0])
metadata = {
    "paper_title": "Llama 2: Open Foundation and Fine-Tuned Chat Models"
}
docs = [Document(text=doc_text, metadata=metadata)]

print(docs[0].get_content())

from llama_index.callbacks import CallbackManager

callback_manager = CallbackManager([])

gpt_35_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo-0613", temperature=0.3),
    callback_manager=callback_manager,
)
gpt_4_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4-0613", temperature=0.3),
    callback_manager=callback_manager,
)

# ## Generate Dataset

from llama_index.evaluation import DatasetGenerator
from llama_index.node_parser import SentenceSplitter

# try evaluation modules
from llama_index.evaluation import RelevancyEvaluator, FaithfulnessEvaluator
from llama_index import PromptTemplate

node_parser = SentenceSplitter()
nodes = node_parser.get_nodes_from_documents(docs)

from tqdm.notebook import tqdm
import json

num_questions_per_chunk = 10
question_gen_query = (
    "You are a Teacher/ Professor. Your task is to setup a quiz/examination."
    f" Using the provided context, formulate {num_questions_per_chunk} that"
    " captures an important fact from the context. \nYou MUST obey the"
    " following criteria:\n- Restrict the question to the context information"
    " provided.\n- Do NOT create a question that cannot be answered from the"
    " context.\n- Phrase the question so that it does NOT refer to specific"
    ' context. For instance, do NOT put phrases like "given provided context"'
    ' or "in this work" in the question, because if the question is asked'
    " elsewhere it wouldn't be provided specific context. Replace these"
    " terms with specific details.\nBAD questions:\nWhat did the author do in"
    " his childhood\nWhat were the main findings in this report\n\nGOOD"
    " questions:\nWhat did Barack Obama do in his childhood\nWhat were the"
    " main findings in the original Transformers paper by Vaswani et"
    " al.\n\nGenerate the questions below:\n"
)

# go through each node one at a time -
# generate questions, filter using eval modules, and dump to file

fp = open("data/qa_pairs.jsonl", "w")
for idx, node in enumerate(nodes):
    dataset_generator = DatasetGenerator(
        [node],
        question_gen_query=question_gen_query,
        service_context=gpt_4_context,
        metadata_mode="all",
    )
    node_questions_0 = dataset_generator.generate_questions_from_nodes(num=10)
    print(f"[Node {idx}] Generated questions:\n {node_questions_0}")
    # for each question, get a response
    for question in tqdm(node_questions_0):
        index = SummaryIndex([node], service_context=gpt_35_context)
        query_engine = index.as_query_engine()
        response = query_engine.query(question)
        out_dict = {"query": question, "response": str(response)}
        print(f"[Node {idx}] Outputs: {out_dict}")
        fp.write(json.dumps(out_dict) + "\n")

fp.close()

# ### Filter out questions using RelevancyEvaluator
# 
# Do a second pass to make sure only questions that can be answerd by context make it into the training set.

# try evaluation modules
from llama_index.evaluation import RelevancyEvaluator, FaithfulnessEvaluator
from llama_index import PromptTemplate
from llama_index.llms import OpenAI

query_eval_tmpl = PromptTemplate(
    "Your task is to evaluate the following: If the response for the query"
    " isn't able to answer the question provided.\nIf query isn't able to"
    " answer the question, answer NO.\nOtherwise answer YES.\nTo elaborate,"
    " you might get an answer like the following: 'The context does not"
    " contain the answer to this question.'Please return NO in that case. You"
    " be given the query and response. Return YES or NO as the answer.\nQuery:"
    " \n {query_str}\nResponse: \n {response_str}\nAnswer: "
)

eval_llm = OpenAI(model="gpt-4-0613")

def filter_data(path: str, out_path: str):
    fp = open(path, "r")
    out_fp = open(out_path, "w")
    new_lines = []
    for idx, line in enumerate(fp):
        qa_pair = json.loads(line)
        eval = eval_llm.complete(
            query_eval_tmpl.format(
                query_str=qa_pair["query"], response_str=qa_pair["response"]
            )
        )

        print(f"[{idx}] QA Pair: {qa_pair} \n Eval: {eval}")
        if "NO" in str(eval):
            continue
        else:
            # new_lines.append(line)
            out_fp.write(line)

filter_data("data/qa_pairs.jsonl", "data/qa_pairs_2.jsonl")

# ### Split into Training and Validation Sets
# 
# We split into training and validation sets.
# 
# **NOTE**: We shuffle the data before splitting. This helps ensure that the training data has coverage throughout the document.

from copy import deepcopy
import random

def split_train_val(
    path: str, out_train_path: str, out_val_path: str, train_split=0.7
):
    with open(path, "r") as fp:
        lines = fp.readlines()

        # shuffle the lines to make sure that the "train questions" cover most fo the context
        shuffled_lines = deepcopy(lines)
        random.shuffle(shuffled_lines)

        split_idx = int(train_split * len(shuffled_lines))
        train_lines = shuffled_lines[:split_idx]
        val_lines = shuffled_lines[split_idx:]
        with open(out_train_path, "w") as out_fp:
            out_fp.write("".join(train_lines))

        with open(out_val_path, "w") as out_fp:
            out_fp.write("".join(val_lines))

split_train_val(
    "data/qa_pairs_2.jsonl",
    "data/qa_pairs_train.jsonl",
    "data/qa_pairs_val.jsonl",
)

# ### Format into Training Data
# 
# Format into training data for OpenAI's finetuning endpoints.
# 
# **NOTE**: We don't use our `OpenAIFinetuningHandler` because that logs the full input prompt including context as the user message. Here we just want to log the query as the user message, because we want to fine-tune gpt-3.5-turbo to "bake in knowledge" into the fine-tuned model.

fp = open("data/qa_pairs_train.jsonl", "r")
out_fp = open("data/qa_pairs_openai.jsonl", "w")
# TODO: try with different system prompts
system_prompt = {
    "role": "system",
    "content": (
        "You are a helpful assistant helping to answer questions about the"
        " Llama 2 paper."
    ),
}
for line in fp:
    qa_pair = json.loads(line)
    user_prompt = {"role": "user", "content": qa_pair["query"]}
    assistant_prompt = {"role": "assistant", "content": qa_pair["response"]}
    out_dict = {
        "messages": [system_prompt, user_prompt, assistant_prompt],
    }
    out_fp.write(json.dumps(out_dict) + "\n")

# ## Fine-tune the Model

from llama_index.finetuning import OpenAIFinetuneEngine

finetune_engine = OpenAIFinetuneEngine(
    "gpt-3.5-turbo",
    "data/qa_pairs_openai.jsonl",
    # start_job_id="<start-job-id>"  # if you have an existing job, can specify id here
)

finetune_engine.finetune()

finetune_engine.get_current_job()

ft_model = finetune_engine.get_finetuned_model()

ft_model

# [Optional] use fine-tuned model in RAG system too
from llama_index import ServiceContext

ft_context = ServiceContext.from_defaults(
    llm=ft_model,
    callback_manager=callback_manager,
)
# baseline RAG system
ft_index = VectorStoreIndex(nodes, service_context=ft_context)
ft_query_engine = ft_index.as_query_engine()

# ## Evaluate Results
# 
# We run evaluations, over both the validation set but also the training set.
# 
# **Wait, isn't evaluating over the training set cheating?**
# 
# - It's a sanity check of how much the model was able to memorize information it's trained on.
# - The training data contains quite a bit of content about the paper, so by answering the training set well the model would at least be well-equipped to answer some questions.

from llama_index.llms import ChatMessage

def load_data(path: str):
    fp = open(path, "r")
    data_dicts = []
    for line in fp:
        d = json.loads(line)
        data_dicts.append(d)
    return data_dicts

train_dicts = load_data("data/qa_pairs_train.jsonl")
eval_dicts = load_data("data/qa_pairs_val.jsonl")

def query_model(model, d):
    # print(d)
    msgs = [
        ChatMessage(
            role="system",
            content=(
                "You are a helpful assistant helping to answer questions about"
                " the Llama 2 paper."
            ),
        ),
        ChatMessage(role="user", content=d["query"]),
    ]

    # try ft-model
    response = model.chat(msgs)
    return str(response)

response = query_model(ft_model, eval_dicts[7])
print(eval_dicts[7])
print(response)

query_model(ft_model, train_dicts[7])
print(train_dicts[7])
print(response)

# ### Setup Baseline RAG system to benchmark
# 
# We setup a baseline RAG system powered by gpt-3.5-turbo to help benchmark the quality of results.

# baseline RAG system
base_index = VectorStoreIndex(nodes, service_context=gpt_35_context)
base_query_engine = base_index.as_query_engine()

# baseline model
base_model = OpenAI(model="gpt-4", temperature=0.3)

query_model(base_model, eval_dicts[80])

# ### Run Evaluations
# 
# We log the responses from the fine-tuned model, the baseline RAG system, and the baseline model.
# 
# We then run all responses through a GPT-4 prompt, comparing each against the ground-truth to measure validity of the result.

import pandas as pd
from tqdm.notebook import tqdm

EVAL_PROMPT_TMPL = PromptTemplate(
    """\
We provide a question and the 'ground-truth' answer. We also provide \
the predicted answer.

Evaluate whether the predicted answer is correct, given its similarity \
to the ground-truth. If details provided in predicted answer are reflected \
in the ground-truth answer, return "YES". To return "YES", the details don't \
need to exactly match. Be lenient in evaluation if the predicted answer \
is missing a few details. Try to make sure that there are no blatant mistakes. \
Otherwise, return "NO".

Question: {question}
Ground-truth Answer: {gt_answer}
Predicted Answer: {pred_answer}
Evaluation Result: \
"""
)

def eval_match_gt(query, gt_response, pred_response):
    llm = OpenAI(model="gpt-4", temperature=0.0)
    fmt_prompt = EVAL_PROMPT_TMPL.format(
        question=query,
        gt_answer=gt_response,
        pred_answer=pred_response,
    )
    result = llm.complete(fmt_prompt)
    if "yes" in str(result).lower():
        return 1
    else:
        return 0

def run_evals(eval_dicts):
    """Run evals - fine-tuned model, RAG system, and base model."""

    raw_responses = []
    for eval_dict in tqdm(eval_dicts):
        gt_response = eval_dict["response"]
        ft_rag_response = str(ft_query_engine.query(eval_dict["query"]))
        ft_response = str(query_model(ft_model, eval_dict))
        rag_response = str(base_query_engine.query(eval_dict["query"]))
        base_response = str(query_model(base_model, eval_dict))

        # try evaluations
        ft_rag_eval = eval_match_gt(
            eval_dict["query"], gt_response, ft_rag_response
        )
        ft_eval = eval_match_gt(eval_dict["query"], gt_response, ft_response)
        rag_eval = eval_match_gt(eval_dict["query"], gt_response, rag_response)
        base_eval = eval_match_gt(
            eval_dict["query"], gt_response, base_response
        )

        response_dict = {
            "query": eval_dict["query"],
            "gt_response": gt_response,
            "ft_rag_response": ft_rag_response,
            "ft_response": ft_response,
            "rag_response": rag_response,
            "base_response": base_response,
            "ft_rag_eval": ft_rag_eval,
            "ft_eval": ft_eval,
            "rag_eval": rag_eval,
            "base_eval": base_eval,
        }

        raw_responses.append(response_dict)

    raw_responses_df = pd.DataFrame(raw_responses)

    eval_dict = {
        "ft_rag_score": raw_responses_df["ft_rag_eval"].mean(),
        "ft_score": raw_responses_df["ft_eval"].mean(),
        "rag_score": raw_responses_df["rag_eval"].mean(),
        "base_score": raw_responses_df["base_eval"].mean(),
    }

    sub_responses_df = raw_responses_df[
        [
            "query",
            "gt_response",
            "ft_rag_response",
            "ft_response",
            "rag_response",
            "base_response",
        ]
    ]

    return eval_dict, raw_responses_df, sub_responses_df

pd.set_option("#display.max_colwidth", None)

# #### Qualitative Evaluations
# 
# Here we show some qualitative output examples over both the training and validation sets.

eval_dict, raw_response_df, sub_responses_df = run_evals(train_dicts[7:8])
#display(eval_dict)
#display(sub_responses_df)

eval_dict, raw_response_df, sub_responses_df = run_evals(eval_dicts[6:7])
#display(eval_dict)
#display(sub_responses_df)

# #### Quantitative Evaluations
# 
# Here we show quantitative metrics over both the training and eval set.

import random

k = 40

train_dicts_sample = random.sample(train_dicts, k)
eval_dicts_sample = random.sample(eval_dicts, k)

result_train, raw_response_df, sub_responses_df = run_evals(train_dicts_sample)
#display(result_train)
# #display(raw_response_df)

# look at where ft_rag_score did well but rag didn't
d = raw_response_df
d[(d["ft_rag_eval"] == 1) & (d["rag_eval"] == 0)]

result_eval, raw_response_df, sub_responses_df = run_evals(eval_dicts_sample)
#display(result_eval)
# #display(raw_response_df)

