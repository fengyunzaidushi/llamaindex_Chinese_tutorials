#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/finetuning/knowledge/finetune_retrieval_aug.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 

# # Fine-tuning with Retrieval Augmentation
# 
# Here we try fine-tuning an LLM with retrieval augmentation, as referenced from the RA-DIT paper: https://arxiv.org/abs/2310.01352.
# 
# For a given (query, response) input/output example, we retrieve the k text chunks with a retriever (the quality of the retriever doesn't have to be perfect, and in fact can be primitive). We then format each query with individually retrieved context, to create k examples (query + context_i, response) for fine-tuning.
# 
# The idea is to allow the LLM to better use background knowledge to synthesize a correct answer, or to synthesize a correct answer even in the absence of good background knowledge. This will enable the LLM to reason from its priors a bit better.

import os
import openai
from llama_index import ServiceContext
from llama_index.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

# ## Setup + Load Data

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

# ### Get Nodes, Setup Vector Index

from llama_index.node_parser import SentenceSplitter
from llama_index import VectorStoreIndex

node_parser = SentenceSplitter()
nodes = node_parser.get_nodes_from_documents(docs)

vector_index = VectorStoreIndex(nodes)

# ## Generate Dataset

from llama_index.evaluation import (
    DatasetGenerator,
    QueryResponseDataset,
)

eval_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4", temperature=0), callback_manager=callback_manager
)
dataset_generator = DatasetGenerator(
    nodes[:39],
    service_context=eval_context,
    show_progress=True,
    num_questions_per_chunk=20,
)

eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=60)

eval_dataset.save_json("data_rag/qa_pairs.json")

# optional
eval_dataset = QueryResponseDataset.from_json("data_rag/qa_pairs.json")

# #### Option 2: Load from existing data 
# 
# If you were already using the fine-tuning knowledge notebook, you can use that instead. 

import json

# load data in from .jsonl format
def load_dataset_from_other_nb(path):
    fp = open(path, "r")
    qr_pairs = []
    for line in fp:
        qa_pair = json.loads(line)
        query_str = qa_pair["query"]
        response_str = qa_pair["response"]
        qr_pairs.append((query_str, response_str))

    return qr_pairs

qr_pairs = load_dataset_from_other_nb("data/qa_pairs_2.jsonl")
eval_dataset = QueryResponseDataset.from_qr_pairs(qr_pairs)

eval_dataset

# ### For each Datapoint, Fetch Retrieved Context with a Retriever
# 
# For each (question, response) pair, fetch the top-k context with a retriever.
# 
# For each pair, we create k (question + context_i, response) new pairs, where we format each input with the QA prompt.

from llama_index import VectorStoreIndex
from llama_index.prompts import PromptTemplate

qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

vector_retriever = vector_index.as_retriever(similarity_top_k=1)

from tqdm.notebook import tqdm

def augment_data_with_retrieval(dataset, retriever, separate_context=False):
    data_list = dataset.qr_pairs
    new_data_list = []
    for query_str, response in tqdm(data_list):
        retrieved_nodes = retriever.retrieve(query_str)
        retrieved_txts = [n.get_content() for n in retrieved_nodes]
        if separate_context:
            for retrieved_txt in retrieved_txts:
                fmt_query_str = qa_prompt_tmpl.format(
                    query_str=query_str, context_str=retrieved_txt
                )
                new_data_list.append((fmt_query_str, response))
        else:
            context_str = "\n\n".join(retrieved_txts)
            fmt_query_str = qa_prompt_tmpl.format(
                query_str=query_str, context_str=context_str
            )
            new_data_list.append((fmt_query_str, response))
    return new_data_list

new_qr_pairs = augment_data_with_retrieval(
    eval_dataset, vector_retriever, separate_context=False
)
new_eval_dataset = QueryResponseDataset.from_qr_pairs(new_qr_pairs)

new_eval_dataset.save_json("data_rag/qa_pairs_ra.json")

new_eval_dataset = QueryResponseDataset.from_json("data_rag/qa_pairs_ra.json")

# ### Split into Training and Validation Sets
# 
# We split into training and validation sets.
# 
# **NOTE**: We shuffle the data before splitting. This helps ensure that the training data has coverage throughout the document.

from copy import deepcopy
import random

def split_train_val(dataset, train_split=0.7):
    lines = dataset.qr_pairs

    # shuffle the lines to make sure that the "train questions" cover most fo the context
    shuffled_lines = deepcopy(lines)
    random.shuffle(shuffled_lines)

    split_idx = int(train_split * len(shuffled_lines))
    train_lines = shuffled_lines[:split_idx]
    val_lines = shuffled_lines[split_idx:]

    return train_lines, val_lines

train_lines, val_lines = split_train_val(new_eval_dataset, train_split=0.7)

train_dataset = QueryResponseDataset.from_qr_pairs(train_lines)
val_dataset = QueryResponseDataset.from_qr_pairs(val_lines)

train_dataset.save_json("data_rag/qa_pairs_train.json")
val_dataset.save_json("data_rag/qa_pairs_val.json")

train_dataset = QueryResponseDataset.from_json("data_rag/qa_pairs_train.json")
val_dataset = QueryResponseDataset.from_json("data_rag/qa_pairs_val.json")

# ### Format into Training Data
# 
# Format into training data for OpenAI's finetuning endpoints.
# 
# **NOTE**: We don't use our `OpenAIFinetuningHandler` because that logs the full input prompt including context as the user message. Here we just want to log the query as the user message, because we want to fine-tune gpt-3.5-turbo to "bake in knowledge" into the fine-tuned model.

def save_openai_data(dataset, out_path):
    # out_fp = open("data_rag/qa_pairs_openai.jsonl", "w")
    out_fp = open(out_path, "w")
    # TODO: try with different system prompts
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful assistant helping to answer questions about the"
            " Llama 2 paper."
        ),
    }
    train_qr_pairs = dataset.qr_pairs
    for line in train_qr_pairs:
        query, response = line
        user_prompt = {"role": "user", "content": query}
        assistant_prompt = {"role": "assistant", "content": response}
        out_dict = {
            "messages": [system_prompt, user_prompt, assistant_prompt],
        }
        out_fp.write(json.dumps(out_dict) + "\n")

save_openai_data(train_dataset, "data_rag/qa_pairs_openai.jsonl")

# ## Fine-tune the Model

from llama_index.finetuning import OpenAIFinetuneEngine

finetune_engine = OpenAIFinetuneEngine(
    "gpt-3.5-turbo",
    "data_rag/qa_pairs_openai.jsonl",
    # start_job_id="<start-job-id>"  # if you have an existing job, can specify id here
)

finetune_engine.finetune()

finetune_engine.get_current_job()

ft_model = finetune_engine.get_finetuned_model()

ft_model

# Use fine-tuned model in RAG system
from llama_index import ServiceContext

ft_context = ServiceContext.from_defaults(
    llm=ft_model,
    callback_manager=callback_manager,
    system_prompt=(
        "You are a helpful assistant helping to answer questions about the"
        " Llama 2 paper."
    ),
)
# fine-tuned RAG system
ft_query_engine = vector_index.as_query_engine(
    similarity_top_k=1, service_context=ft_context
)

response = ft_query_engine.query(
    "How is the margin component added in the loss of the reward model in"
    " Llama 2?"
)
print(str(response))

base_query_engine = vector_index.as_query_engine(similarity_top_k=1)
base_response = base_query_engine.query(
    "How is the margin component added in the loss of the reward model in"
    " Llama 2?"
)
print(str(base_response))

# ## Evaluate Results
# 
# We run evaluations, over both the validation set but also the training set (as a sanity check)

import nest_asyncio

nest_asyncio.apply()

from llama_index.llms import ChatMessage
from llama_index.evaluation.eval_utils import get_responses, get_results_df
from llama_index.evaluation import BatchEvalRunner

# train_dataset = QueryResponseDataset.from_json("data_rag/qa_pairs_train.json")
# val_dataset = QueryResponseDataset.from_json("data_rag/qa_pairs_val.json")

# Load dataset
# NOTE: we need to run over the original questions, not the retrieval-augmented questions.
# Since our query engines will perform retrieval augmentation under the hood!

# TODO: have better code here
qr_pairs = load_dataset_from_other_nb("data/qa_pairs_2.jsonl")
eval_dataset = QueryResponseDataset.from_qr_pairs(qr_pairs)

# evaluate over training dataset for now
sample_size = 50

eval_qs = eval_dataset.questions[:sample_size]
ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs[:sample_size]]

pred_responses = get_responses(eval_qs, ft_query_engine, show_progress=True)

base_pred_responses = get_responses(
    eval_qs, base_query_engine, show_progress=True
)

import numpy as np

pred_response_strs = [str(p) for p in pred_responses]
base_pred_response_strs = [str(p) for p in base_pred_responses]

from llama_index.evaluation import (
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
)

eval_service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4"))
# NOTE: can uncomment other evaluators
evaluator_c = CorrectnessEvaluator(service_context=eval_service_context)
evaluator_s = SemanticSimilarityEvaluator(service_context=eval_service_context)

evaluator_dict = {
    "correctness": evaluator_c,
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
    ["RAG Fine-tuned LLM", "Base LLM"],
    ["correctness", "semantic_similarity"],
)
#display(results_df)

