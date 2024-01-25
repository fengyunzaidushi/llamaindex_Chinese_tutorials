#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/retrievers/recursive_retriever_nodes.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Recursive Retriever + Node References
# 
# This guide shows how you can use recursive retrieval to traverse node relationships and fetch nodes based on "references".
# 
# Node references are a powerful concept. When you first perform retrieval, you may want to retrieve the reference as opposed to the raw text. You can have multiple references point to the same node.
# 

# - **Chunk references**: Different chunk sizes referring to a bigger chunk
# - **Metadata references**: Summaries + Generated Questions referring to a bigger chunk

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('env', 'OPENAI_API_KEY=YOUR_API_KEY')

get_ipython().run_line_magic('pip', 'install -U llama_hub llama_index braintrust autoevals pypdf pillow transformers torch torchvision')

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## Load Data + Setup
# 

#("wget 'data/'")
#('wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"')

from pathlib import Path
from llama_hub.file.pdf.base import PDFReader
from llama_index.response.notebook_utils import #display_source_node
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
import json

loader = PDFReader()
docs0 = loader.load_data(file=Path("./data/llama2.pdf"))

from llama_index import Document

doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]

from llama_index.node_parser import SentenceSplitter
from llama_index.schema import IndexNode

node_parser = SentenceSplitter(chunk_size=1024)

base_nodes = node_parser.get_nodes_from_documents(docs)
# set node ids to be a constant
for idx, node in enumerate(base_nodes):
    node.id_ = f"node-{idx}"

from llama_index.embeddings import resolve_embed_model

embed_model = resolve_embed_model("local:BAAI/bge-small-en")
llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model
)

# ## Baseline Retriever
# 
# Define a baseline retriever that simply fetches the top-k raw text nodes by embedding similarity.

base_index = VectorStoreIndex(base_nodes, service_context=service_context)
base_retriever = base_index.as_retriever(similarity_top_k=2)

retrievals = base_retriever.retrieve(
    "Can you tell me about the key concepts for safety finetuning"
)

for n in retrievals:
    #display_source_node(n, source_length=1500)

query_engine_base = RetrieverQueryEngine.from_args(
    base_retriever, service_context=service_context
)

response = query_engine_base.query(
    "Can you tell me about the key concepts for safety finetuning"
)
print(str(response))

# ## Chunk References: Smaller Child Chunks Referring to Bigger Parent Chunk
# 

# 
# During query-time, we retrieve smaller chunks, but we follow references to bigger chunks. This allows us to have more context for synthesis.

sub_chunk_sizes = [128, 256, 512]
sub_node_parsers = [SentenceSplitter(chunk_size=c) for c in sub_chunk_sizes]

all_nodes = []
for base_node in base_nodes:
    for n in sub_node_parsers:
        sub_nodes = n.get_nodes_from_documents([base_node])
        sub_inodes = [
            IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
        ]
        all_nodes.extend(sub_inodes)

    # also add original node to node
    original_node = IndexNode.from_text_node(base_node, base_node.node_id)
    all_nodes.append(original_node)

all_nodes_dict = {n.node_id: n for n in all_nodes}

vector_index_chunk = VectorStoreIndex(
    all_nodes, service_context=service_context
)

vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=2)

retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True,
)

nodes = retriever_chunk.retrieve(
    "Can you tell me about the key concepts for safety finetuning"
)
for node in nodes:
    #display_source_node(node, source_length=2000)

query_engine_chunk = RetrieverQueryEngine.from_args(
    retriever_chunk, service_context=service_context
)

response = query_engine_chunk.query(
    "Can you tell me about the key concepts for safety finetuning"
)
print(str(response))

# ## Metadata References: Summaries + Generated Questions referring to a bigger chunk
# 

# 
# This additional context includes summaries as well as generated questions.
# 
# During query-time, we retrieve smaller chunks, but we follow references to bigger chunks. This allows us to have more context for synthesis.

from llama_index.node_parser import SentenceSplitter
from llama_index.schema import IndexNode
from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)

extractors = [
    SummaryExtractor(summaries=["self"], show_progress=True),
    QuestionsAnsweredExtractor(questions=5, show_progress=True),
]

# run metadata extractor across base nodes, get back dictionaries
metadata_dicts = []
for extractor in extractors:
    metadata_dicts.extend(extractor.extract(base_nodes))

# cache metadata dicts
def save_metadata_dicts(path):
    with open(path, "w") as fp:
        for m in metadata_dicts:
            fp.write(json.dumps(m) + "\n")

def load_metadata_dicts(path):
    with open(path, "r") as fp:
        metadata_dicts = [json.loads(l) for l in fp.readlines()]
        return metadata_dicts

save_metadata_dicts("data/llama2_metadata_dicts.jsonl")

metadata_dicts = load_metadata_dicts("data/llama2_metadata_dicts.jsonl")

# all nodes consists of source nodes, along with metadata
import copy

all_nodes = copy.deepcopy(base_nodes)
for idx, d in enumerate(metadata_dicts):
    inode_q = IndexNode(
        text=d["questions_this_excerpt_can_answer"],
        index_id=base_nodes[idx].node_id,
    )
    inode_s = IndexNode(
        text=d["section_summary"], index_id=base_nodes[idx].node_id
    )
    all_nodes.extend([inode_q, inode_s])

all_nodes_dict = {n.node_id: n for n in all_nodes}

## Load index into vector index
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

vector_index_metadata = VectorStoreIndex(
    all_nodes, service_context=service_context
)

vector_retriever_metadata = vector_index_metadata.as_retriever(
    similarity_top_k=2
)

retriever_metadata = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_metadata},
    node_dict=all_nodes_dict,
    verbose=True,
)

nodes = retriever_metadata.retrieve(
    "Can you tell me about the key concepts for safety finetuning"
)
for node in nodes:
    #display_source_node(node, source_length=2000)

query_engine_metadata = RetrieverQueryEngine.from_args(
    retriever_metadata, service_context=service_context
)

response = query_engine_metadata.query(
    "Can you tell me about the key concepts for safety finetuning"
)
print(str(response))

# ## Evaluation
# 
# We evaluate how well our recursive retrieval + node reference methods work. We evaluate both chunk references as well as metadata references. We use embedding similarity lookup to retrieve the reference nodes.
# 
# We compare both methods against a baseline retriever where we fetch the raw nodes directly.
# 

# ### Dataset Generation
# 
# We first generate a dataset of questions from the set of text chunks.

from llama_index.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
import nest_asyncio

nest_asyncio.apply()

eval_dataset = generate_question_context_pairs(base_nodes)

eval_dataset.save_json("data/llama2_eval_dataset.json")

# optional
eval_dataset = EmbeddingQAFinetuneDataset.from_json(
    "data/llama2_eval_dataset.json"
)

# ### Compare Results
# 
# We run evaluations on each of the retrievers to measure hit rate and MRR.
# 
# We find that retrievers with node references (either chunk or metadata) tend to perform better than retrieving the raw chunks.

import pandas as pd
from llama_index.evaluation import RetrieverEvaluator, get_retrieval_results_df

# set vector retriever similarity top k to higher
top_k = 10

def #display_results(names, results_arr):
    """Display results from evaluate."""

    hit_rates = []
    mrrs = []
    for name, eval_results in zip(names, results_arr):
        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)
        results_df = pd.DataFrame(metric_dicts)

        hit_rate = results_df["hit_rate"].mean()
        mrr = results_df["mrr"].mean()
        hit_rates.append(hit_rate)
        mrrs.append(mrr)

    final_df = pd.DataFrame(
        {"retrievers": names, "hit_rate": hit_rates, "mrr": mrrs}
    )
    #display(final_df)

vector_retriever_chunk = vector_index_chunk.as_retriever(
    similarity_top_k=top_k
)
retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True,
)
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever_chunk
)
# try it out on an entire dataset
results_chunk = await retriever_evaluator.aevaluate_dataset(
    eval_dataset, show_progress=True
)

vector_retriever_metadata = vector_index_metadata.as_retriever(
    similarity_top_k=top_k
)
retriever_metadata = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_metadata},
    node_dict=all_nodes_dict,
    verbose=True,
)
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever_metadata
)
# try it out on an entire dataset
results_metadata = await retriever_evaluator.aevaluate_dataset(
    eval_dataset, show_progress=True
)

base_retriever = base_index.as_retriever(similarity_top_k=10)
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=base_retriever
)
# try it out on an entire dataset
results_base = await retriever_evaluator.aevaluate_dataset(
    eval_dataset, show_progress=True
)

full_results_df = get_retrieval_results_df(
    [
        "Base Retriever",
        "Retriever (Chunk References)",
        "Retriever (Metadata References)",
    ],
    [results_base, results_chunk, results_metadata],
)
#display(full_results_df)

