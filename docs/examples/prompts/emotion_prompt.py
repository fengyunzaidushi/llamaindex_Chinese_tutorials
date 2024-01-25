#!/usr/bin/env python
# coding: utf-8

# # EmotionPrompt in RAG
# 

# Emotional Stimuli](https://arxiv.org/pdf/2307.11760.pdf)" by Li et al., in this guide we show you how to evaluate the effects of emotional stimuli on your RAG pipeline:
# 
# 1. Setup the RAG pipeline with a basic vector index with the core QA template.
# 2. Create some candidate stimuli (inspired by Fig. 2 of the paper)
# 3. For each candidate stimulit, prepend to QA prompt and evaluate.
# 

import nest_asyncio

nest_asyncio.apply()

# ## Setup Data
# 
# We use the Llama 2 paper as the input data source for our RAG pipeline.

#('mkdir data && wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"')

from pathlib import Path
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import IndexNode

docs0 = PyMuPDFReader().load(file_path=Path("./data/llama2.pdf"))
doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]
node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
base_nodes = node_parser.get_nodes_from_documents(docs)

# ## Setup Vector Index over this Data
# 
# We load this data into an in-memory vector store (embedded with OpenAI embeddings).
# 
# We'll be aggressively optimizing the QA prompt for this RAG pipeline.

from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

rag_service_context = ServiceContext.from_defaults(llm=llm)

index = VectorStoreIndex(base_nodes, service_context=rag_service_context)

query_engine = index.as_query_engine(similarity_top_k=2)

# ## Evaluation Setup

# #### Golden Dataset

# Here we load in a "golden" dataset.
# 
# **NOTE**: We pull this in from Dropbox. For details on how to generate a dataset please see our `DatasetGenerator` module.

#('wget "https://www.dropbox.com/scl/fi/fh9vsmmm8vu0j50l3ss38/llama2_eval_qr_dataset.json?rlkey=kkoaez7aqeb4z25gzc06ak6kb&dl=1" -O data/llama2_eval_qr_dataset.json')

from llama_index.evaluation import QueryResponseDataset

# optional
eval_dataset = QueryResponseDataset.from_json(
    "data/llama2_eval_qr_dataset.json"
)

# #### Get Evaluator

from llama_index.evaluation.eval_utils import get_responses

from llama_index.evaluation import CorrectnessEvaluator, BatchEvalRunner

eval_service_context = ServiceContext.from_defaults(llm=llm)
evaluator_c = CorrectnessEvaluator(service_context=eval_service_context)
evaluator_dict = {"correctness": evaluator_c}
batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)

# #### Define Correctness Eval Function

import numpy as np

async def get_correctness(query_engine, eval_qa_pairs, batch_runner):
    # then evaluate
    # TODO: evaluate a sample of generated results
    eval_qs = [q for q, _ in eval_qa_pairs]
    eval_answers = [a for _, a in eval_qa_pairs]
    pred_responses = get_responses(eval_qs, query_engine, show_progress=True)

    eval_results = await batch_runner.aevaluate_responses(
        eval_qs, responses=pred_responses, reference=eval_answers
    )
    avg_correctness = np.array(
        [r.score for r in eval_results["correctness"]]
    ).mean()
    return avg_correctness

# ## Try Out Emotion Prompts
# 
# We pul some emotion stimuli from the paper to try out.

emotion_stimuli_dict = {
    "ep01": "Write your answer and give me a confidence score between 0-1 for your answer. ",
    "ep02": "This is very important to my career. ",
    "ep03": "You'd better be sure.",
    # add more from the paper here!!
}

# NOTE: ep06 is the combination of ep01, ep02, ep03
emotion_stimuli_dict["ep06"] = (
    emotion_stimuli_dict["ep01"]
    + emotion_stimuli_dict["ep02"]
    + emotion_stimuli_dict["ep03"]
)

# ###

QA_PROMPT_KEY = "response_synthesizer:text_qa_template"

from llama_index.prompts import PromptTemplate

qa_tmpl_str = """\
Context information is below. 
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, \
answer the query.
{emotion_str}
Query: {query_str}
Answer: \
"""
qa_tmpl = PromptTemplate(qa_tmpl_str)

# #### Prepend emotions 

async def run_and_evaluate(
    query_engine, eval_qa_pairs, batch_runner, emotion_stimuli_str, qa_tmpl
):
    """Run and evaluate."""
    new_qa_tmpl = qa_tmpl.partial_format(emotion_str=emotion_stimuli_str)

    old_qa_tmpl = query_engine.get_prompts()[QA_PROMPT_KEY]
    query_engine.update_prompts({QA_PROMPT_KEY: new_qa_tmpl})
    avg_correctness = await get_correctness(
        query_engine, eval_qa_pairs, batch_runner
    )
    query_engine.update_prompts({QA_PROMPT_KEY: old_qa_tmpl})
    return avg_correctness

# try out ep01
correctness_ep01 = await run_and_evaluate(
    query_engine,
    eval_dataset.qr_pairs,
    batch_runner,
    emotion_stimuli_dict["ep01"],
    qa_tmpl,
)

print(correctness_ep01)

# try out ep02
correctness_ep02 = await run_and_evaluate(
    query_engine,
    eval_dataset.qr_pairs,
    batch_runner,
    emotion_stimuli_dict["ep02"],
    qa_tmpl,
)

print(correctness_ep02)

# try none
correctness_base = await run_and_evaluate(
    query_engine, eval_dataset.qr_pairs, batch_runner, "", qa_tmpl
)

print(correctness_base)

