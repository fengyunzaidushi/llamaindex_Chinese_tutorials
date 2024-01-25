#!/usr/bin/env python
# coding: utf-8

# # Time-Weighted Rerank
# 
# Showcase capabilities of time-weighted node postprocessor

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.postprocessor import (
    TimeWeightedPostprocessor,
)
from llama_index.text_splitter import SentenceSplitter
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.response.notebook_utils import #display_response
from datetime import datetime, timedelta

# ### Parse Documents into Nodes, add to Docstore
# 

# for one specific section, which details the amount of funding they raised for Viaweb. 
# 
# V1: 50k, V2: 30k, V3: 10K
# 
# V1: -1 day, V2: -2 days, V3: -3 days
# 
# The idea is to encourage index to fetch the most recent info (which is V3)

# load documents
from llama_index.storage.storage_context import StorageContext

now = datetime.now()
key = "__last_accessed__"

doc1 = SimpleDirectoryReader(
    input_files=["./test_versioned_data/paul_graham_essay_v1.txt"]
).load_data()[0]

doc2 = SimpleDirectoryReader(
    input_files=["./test_versioned_data/paul_graham_essay_v2.txt"]
).load_data()[0]

doc3 = SimpleDirectoryReader(
    input_files=["./test_versioned_data/paul_graham_essay_v3.txt"]
).load_data()[0]

# define service context (wrapper container around current classes)
text_splitter = SentenceSplitter(chunk_size=512)
service_context = ServiceContext.from_defaults(text_splitter=text_splitter)

# use node parser in service context to parse docs into nodes
nodes1 = text_splitter.get_nodes_from_documents([doc1])
nodes2 = text_splitter.get_nodes_from_documents([doc2])
nodes3 = text_splitter.get_nodes_from_documents([doc3])

# fetch the modified chunk from each document, set metadata
# also exclude the date from being read by the LLM
nodes1[14].metadata[key] = (now - timedelta(hours=3)).timestamp()
nodes1[14].excluded_llm_metadata_keys = [key]

nodes2[14].metadata[key] = (now - timedelta(hours=2)).timestamp()
nodes2[14].excluded_llm_metadata_keys = [key]

nodes3[14].metadata[key] = (now - timedelta(hours=1)).timestamp()
nodes2[14].excluded_llm_metadata_keys = [key]

# add to docstore
docstore = SimpleDocumentStore()
nodes = [nodes1[14], nodes2[14], nodes3[14]]
docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

# ### Build Index

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)

# ### Define Recency Postprocessors

node_postprocessor = TimeWeightedPostprocessor(
    time_decay=0.5, time_access_refresh=False, top_k=1
)

# ### Query Index

# naive query
query_engine = index.as_query_engine(
    similarity_top_k=3,
)
response = query_engine.query(
    "How much did the author raise in seed funding from Idelle's husband"
    " (Julian) for Viaweb?",
)

#display_response(response)

# query using time weighted node postprocessor

query_engine = index.as_query_engine(
    similarity_top_k=3, node_postprocessors=[node_postprocessor]
)
response = query_engine.query(
    "How much did the author raise in seed funding from Idelle's husband"
    " (Julian) for Viaweb?",
)

#display_response(response)

# ### Query Index (Lower-Level Usage)
# 

# finally synthesize response through a summary index.

from llama_index import SummaryIndex

query_str = (
    "How much did the author raise in seed funding from Idelle's husband"
    " (Julian) for Viaweb?"
)

query_engine = index.as_query_engine(
    similarity_top_k=3, response_mode="no_text"
)
init_response = query_engine.query(
    query_str,
)
resp_nodes = [n for n in init_response.source_nodes]

# get the post-processed nodes -- which should be the top-1 sorted by date
new_resp_nodes = node_postprocessor.postprocess_nodes(resp_nodes)

summary_index = SummaryIndex([n.node for n in new_resp_nodes])
query_engine = summary_index.as_query_engine()
response = query_engine.query(query_str)

#display_response(response)

