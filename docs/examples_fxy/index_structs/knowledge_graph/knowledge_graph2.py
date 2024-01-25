#!/usr/bin/env python
# coding: utf-8

# # Knowledge Graph Construction w/ WikiData Filtering
# 

# 
# This is a simplified version, find out more about using wikipedia for filtering, check here
# - [Make Meaningful Knowledge Graph from OpenSource REBEL Model](https://medium.com/@haiyangli_38602/make-meaningful-knowledge-graph-from-opensource-rebel-model-6f9729a55527)

# ## Setup

#('pip install llama_index transformers wikipedia html2text pyvis')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    ServiceContext,
    KnowledgeGraphIndex,
)
from llama_index import SimpleWebPageReader
from llama_index.graph_stores import SimpleGraphStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import OpenAI

# ## 1. extract via huggingface pipeline
# 
# The initial pipeline uses the provided extraction code from the [HuggingFace model card](https://huggingface.co/Babelscape/rebel-large).

from transformers import pipeline

triplet_extractor = pipeline(
    "text2text-generation",
    model="Babelscape/rebel-large",
    tokenizer="Babelscape/rebel-large",
    # comment this line to run on CPU
    device="cuda:0",
)

def extract_triplets(input_text):
    text = triplet_extractor.tokenizer.batch_decode(
        [
            triplet_extractor(
                input_text, return_tensors=True, return_text=False
            )[0]["generated_token_ids"]
        ]
    )[0]

    triplets = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"
    for token in (
        text.replace("<s>", "")
        .replace("<pad>", "")
        .replace("</s>", "")
        .split()
    ):
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append(
                    (subject.strip(), relation.strip(), object_.strip())
                )
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append(
                    (subject.strip(), relation.strip(), object_.strip())
                )
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token

    if subject != "" and relation != "" and object_ != "":
        triplets.append((subject.strip(), relation.strip(), object_.strip()))

    return triplets

# ## 2. Extract with wiki filtering
# 
# Optionally, we can filter our extracted relations using data from wikipedia.
# 
# 

import wikipedia

class WikiFilter:
    def __init__(self):
        self.cache = {}

    def filter(self, candidate_entity):
        # check the cache to avoid network calls
        if candidate_entity in self.cache:
            return self.cache[candidate_entity]["title"]

        # pull the page from wikipedia -- if it exists
        try:
            page = wikipedia.page(candidate_entity, auto_suggest=False)
            entity_data = {
                "title": page.title,
                "url": page.url,
                "summary": page.summary,
            }

            # cache the page title and original entity
            self.cache[candidate_entity] = entity_data
            self.cache[page.title] = entity_data

            return entity_data["title"]
        except:
            return None

wiki_filter = WikiFilter()

def extract_triplets_wiki(text):
    relations = extract_triplets(text)

    filtered_relations = []
    for relation in relations:
        (subj, rel, obj) = relation
        filtered_subj = wiki_filter.filter(subj)
        filtered_obj = wiki_filter.filter(obj)

        # skip if at least one entity not linked to wiki
        if filtered_subj is None and filtered_obj is None:
            continue

        filtered_relations.append(
            (
                filtered_subj or subj,
                rel,
                filtered_obj or obj,
            )
        )

    return filtered_relations

# ## Run with Llama_Index

from llama_index import download_loader

ArxivReader = download_loader("ArxivReader")

loader = ArxivReader()
documents = loader.load_data(
    search_query="Retrieval Augmented Generation", max_results=1
)

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index import Document

# merge all documents into one, since it's split by page
documents = [Document(text="".join([x.text for x in documents]))]

# set up service context
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=256)

# set up graph storage context
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# NOTE: This next cell takes about 4mins on GPU.

index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=3,
    kg_triplet_extract_fn=extract_triplets,
    storage_context=storage_context,
    service_context=service_context,
    include_embeddings=True,
)

index1 = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=3,
    kg_triplet_extract_fn=extract_triplets_wiki,
    storage_context=storage_context,
    service_context=service_context,
    include_embeddings=True,
)

## create graph
from pyvis.network import Network

g = index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.save_graph("non_filtered_graph.html")

from IPython.#display import HTML

HTML(filename="non_filtered_graph.html")

## create graph
from pyvis.network import Network

g = index1.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.save_graph("wiki_filtered_graph.html")

from IPython.#display import HTML

HTML(filename="wiki_filtered_graph.html")

