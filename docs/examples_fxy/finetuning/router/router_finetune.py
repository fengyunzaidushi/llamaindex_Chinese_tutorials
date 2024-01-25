#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/finetuning/router/router_finetune.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Router Fine-tuning
# 

# 
# 1. Fine-tuning embeddings
# 2. Fine-tuning a cross-encoder
# 
# Our dataset will be Wikipedia articles of different cities. 
# 
# We will generate a synthetic dataset for each approach to fine-tune over. We will also run some basic evaluations.

import nest_asyncio

nest_asyncio.apply()

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

#('pip install spacy')

# ## Setup

wiki_titles = [
    "Toronto",
    "Seattle",
    "Chicago",
    "Boston",
    "Houston",
    "Tokyo",
    "Berlin",
    "Lisbon",
]

from pathlib import Path

import requests

for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

from llama_index import SimpleDirectoryReader

# Load all wiki documents
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()

from llama_index import ServiceContext
from llama_index.llms import OpenAI

gpt_35_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3)
)

# define descriptions/choices for tools
city_descs_dict = {}
# these choices will be passed to the router selector
choices = []
choice_to_id_dict = {}

for idx, wiki_title in enumerate(wiki_titles):
    vector_desc = (
        "Useful for questions related to specific aspects of"
        f" {wiki_title} (e.g. the history, arts and culture,"
        " sports, demographics, or more)."
    )
    summary_desc = (
        "Useful for any requests that require a holistic summary"
        f" of EVERYTHING about {wiki_title}. For questions about"
        " more specific sections, please use the vector_tool."
    )
    doc_id_vector = f"{wiki_title}_vector"
    doc_id_summary = f"{wiki_title}_summary"
    city_descs_dict[doc_id_vector] = vector_desc
    city_descs_dict[doc_id_summary] = summary_desc

    choices.extend([vector_desc, summary_desc])
    choice_to_id_dict[idx * 2] = f"{wiki_title}_vector"
    choice_to_id_dict[idx * 2 + 1] = f"{wiki_title}_summary"

from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate

llm = OpenAI(model_name="gpt-3.5-turbo")

summary_q_tmpl = """\
You are a summary question generator. Given an existing question which asks for a summary of a given topic, \
generate {num_vary} related queries that also ask for a summary of the topic.

For example, assuming we're generating 3 related questions:
Base Question: Can you tell me more about Boston?
Question Variations:
Give me an overview of Boston as a city.
Can you describe different aspects of Boston, from the history to the sports scene to the food?
Write a concise summary of Boston; I've never been.

Now let's give it a shot! 

Base Question: {base_question}
Question Variations:
"""
summary_q_prompt = PromptTemplate(summary_q_tmpl)

from collections import defaultdict
from llama_index.evaluation import DatasetGenerator
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from llama_index.node_parser import SimpleNodeParser
from tqdm.notebook import tqdm

def generate_dataset(
    wiki_titles,
    city_descs_dict,
    llm,
    summary_q_prompt,
    num_vector_qs_per_node=2,
    num_summary_qs=4,
):
    # generate dataset from each wikipedia page
    queries = {}
    corpus = {}
    relevant_docs = defaultdict(list)
    for idx, wiki_title in enumerate(tqdm(wiki_titles)):
        doc_id_vector = f"{wiki_title}_vector"
        doc_id_summary = f"{wiki_title}_summary"
        corpus[doc_id_vector] = city_descs_dict[doc_id_vector]
        corpus[doc_id_summary] = city_descs_dict[doc_id_summary]

        # generate questions for semantic search
        node_parser = SimpleNodeParser.from_defaults()
        nodes = node_parser.get_nodes_from_documents(city_docs[wiki_title])

        dataset_generator = DatasetGenerator(
            nodes,
            service_context=gpt_35_context,
            num_questions_per_chunk=num_vector_qs_per_node,
        )
        doc_questions = dataset_generator.generate_questions_from_nodes(
            num=len(nodes) * num_vector_qs_per_node
        )
        for query_idx, doc_question in enumerate(doc_questions):
            query_id = f"{wiki_title}_{query_idx}"
            relevant_docs[query_id] = [doc_id_vector]
            queries[query_id] = doc_question

        # generate questions for summarization
        base_q = f"Give me a summary of {wiki_title}"
        fmt_prompt = summary_q_prompt.format(
            num_vary=num_summary_qs,
            base_question=base_q,
        )
        raw_response = llm.complete(fmt_prompt)
        raw_lines = str(raw_response).split("\n")
        doc_summary_questions = [l for l in raw_lines if l != ""]
        print(f"[{idx}] Original Question: {base_q}")
        print(
            f"[{idx}] Generated Question Variations: {doc_summary_questions}"
        )
        for query_idx, doc_summary_question in enumerate(
            doc_summary_questions
        ):
            query_id = f"{wiki_title}_{query_idx}"
            relevant_docs[query_id] = [doc_id_summary]
            queries[query_id] = doc_summary_question

    return EmbeddingQAFinetuneDataset(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs
    )

dataset = generate_dataset(
    wiki_titles,
    city_descs_dict,
    llm,
    summary_q_prompt,
    num_vector_qs_per_node=4,
    num_summary_qs=5,
)

# dataset.queries

# [optional] save
dataset.save_json("dataset.json")

# [optional] load
dataset = EmbeddingQAFinetuneDataset.from_json("dataset.json")

import random

def split_train_val_by_query(dataset, split=0.7):
    """Split dataset by queries."""
    query_ids = list(dataset.queries.keys())
    query_ids_shuffled = random.sample(query_ids, len(query_ids))
    split_idx = int(len(query_ids) * split)
    train_query_ids = query_ids_shuffled[:split_idx]
    eval_query_ids = query_ids_shuffled[split_idx:]

    train_queries = {qid: dataset.queries[qid] for qid in train_query_ids}
    eval_queries = {qid: dataset.queries[qid] for qid in eval_query_ids}

    train_rel_docs = {
        qid: dataset.relevant_docs[qid] for qid in train_query_ids
    }
    eval_rel_docs = {qid: dataset.relevant_docs[qid] for qid in eval_query_ids}

    train_dataset = EmbeddingQAFinetuneDataset(
        queries=train_queries,
        corpus=dataset.corpus,
        relevant_docs=train_rel_docs,
    )
    eval_dataset = EmbeddingQAFinetuneDataset(
        queries=eval_queries,
        corpus=dataset.corpus,
        relevant_docs=eval_rel_docs,
    )
    return train_dataset, eval_dataset

train_dataset, eval_dataset = split_train_val_by_query(dataset, split=0.7)

# ## Fine-tuning Embeddings
# 

# generate embeddings dataset
from llama_index.finetuning import SentenceTransformersFinetuneEngine

finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="test_model3",
    val_dataset=eval_dataset,
    epochs=30,  # can set to higher (haven't tested)
)

finetune_engine.finetune()

ft_embed_model = finetune_engine.get_finetuned_model()

ft_embed_model

# ## Run Evaluations
# 

# 
# We plug both into our `EmbeddingSelector` abstraction.
# 
# We also compare against a base `LLMSingleSelector` using GPT-4. 

# define baseline embedding model
from llama_index.embeddings import resolve_embed_model

base_embed_model = resolve_embed_model("local:BAAI/bge-small-en")

from llama_index.selectors import EmbeddingSingleSelector, LLMSingleSelector

ft_selector = EmbeddingSingleSelector.from_defaults(embed_model=ft_embed_model)
base_selector = EmbeddingSingleSelector.from_defaults(
    embed_model=base_embed_model
)

import numpy as np

def run_evals(eval_dataset, selector, choices, choice_to_id_dict):
    # we just measure accuracy
    eval_pairs = eval_dataset.query_docid_pairs
    matches = []
    for query, relevant_doc_ids in tqdm(eval_pairs):
        result = selector.select(choices, query)
        # assume single selection for now
        pred_doc_id = choice_to_id_dict[result.inds[0]]
        gt_doc_id = relevant_doc_ids[0]
        matches.append(gt_doc_id == pred_doc_id)
    return np.array(matches)

ft_matches = run_evals(eval_dataset, ft_selector, choices, choice_to_id_dict)

np.mean(ft_matches)

base_matches = run_evals(
    eval_dataset, base_selector, choices, choice_to_id_dict
)

np.mean(base_matches)

# also try LLM
from llama_index.llms import OpenAI

eval_llm = OpenAI(model="gpt-3.5-turbo")

llm_selector = LLMSingleSelector.from_defaults(
    service_context=ServiceContext.from_defaults(llm=eval_llm)
)

llm_matches = run_evals(eval_dataset, llm_selector, choices, choice_to_id_dict)

np.mean(llm_matches)

import pandas as pd

eval_df = pd.DataFrame(
    {
        "Base embedding model": np.mean(base_matches),
        "GPT-3.5": np.mean(llm_matches),
        "Fine-tuned embedding model": np.mean(ft_matches),
    },
    index=["Match Rate"],
)
#display(eval_df)

# ## Plug into Router
# 
# We plug this into our `RouterQueryEngine` as an `EmbeddingSelector` (by default, an `LLMSingleSelector` is used in our router query engine).

from llama_index.query_engine import RouterQueryEngine
from llama_index import SummaryIndex, VectorStoreIndex
from llama_index.tools.query_engine import QueryEngineTool

# define indexes/tools for wikipedia entries
tools = []
for idx, wiki_title in enumerate(tqdm(wiki_titles)):
    doc_id_vector = f"{wiki_title}_vector"
    doc_id_summary = f"{wiki_title}_summary"

    vector_index = VectorStoreIndex.from_documents(city_docs[wiki_title])
    summary_index = SummaryIndex.from_documents(city_docs[wiki_title])
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(),
        description=city_descs_dict[doc_id_vector],
    )
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(),
        description=city_descs_dict[doc_id_summary],
    )
    tools.extend([vector_tool, summary_tool])

router_query_engine = RouterQueryEngine.from_defaults(
    selector=ft_selector.from_defaults(), query_engine_tools=tools
)

response = router_query_engine.query(
    "Tell me more about the sports teams in Toronto"
)

print(str(response))

response.source_nodes[0].get_content()

