#!/usr/bin/env python
# coding: utf-8

# # Recency Filtering
# 
# Showcase capabilities of recency-weighted node postprocessor

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.postprocessor import (
    FixedRecencyPostprocessor,
    EmbeddingRecencyPostprocessor,
)
from llama_index.text_splitter import SentenceSplitter
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.response.notebook_utils import #display_response

# ### Parse Documents into Nodes, add to Docstore
# 

# for one specific section, which details the amount of funding they raised for Viaweb. 
# 
# V1: 50k, V2: 30k, V3: 10K
# 
# V1: 2020-01-01, V2: 2020-02-03, V3: 2022-04-12
# 
# The idea is to encourage index to fetch the most recent info (which is V3)

# load documents
from llama_index.storage.storage_context import StorageContext

def get_file_metadata(file_name: str):
    """Get file metadata."""
    if "v1" in file_name:
        return {"date": "2020-01-01"}
    elif "v2" in file_name:
        return {"date": "2020-02-03"}
    elif "v3" in file_name:
        return {"date": "2022-04-12"}
    else:
        raise ValueError("invalid file")

documents = SimpleDirectoryReader(
    input_files=[
        "test_versioned_data/paul_graham_essay_v1.txt",
        "test_versioned_data/paul_graham_essay_v2.txt",
        "test_versioned_data/paul_graham_essay_v3.txt",
    ],
    file_metadata=get_file_metadata,
).load_data()

# define service context (wrapper container around current classes)
text_splitter = SentenceSplitter(chunk_size=512)
service_context = ServiceContext.from_defaults(text_splitter=text_splitter)

# use node parser to parse into nodes
nodes = text_splitter.get_nodes_from_documents(documents)

# add to docstore
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

print(documents[2].get_text())

# ### Build Index

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)

# ### Define Recency Postprocessors

node_postprocessor = FixedRecencyPostprocessor(service_context=service_context)

node_postprocessor_emb = EmbeddingRecencyPostprocessor(
    service_context=service_context
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

# query using fixed recency node postprocessor

query_engine = index.as_query_engine(
    similarity_top_k=3, node_postprocessors=[node_postprocessor]
)
response = query_engine.query(
    "How much did the author raise in seed funding from Idelle's husband"
    " (Julian) for Viaweb?",
)

# query using embedding-based node postprocessor

query_engine = index.as_query_engine(
    similarity_top_k=3, node_postprocessors=[node_postprocessor_emb]
)
response = query_engine.query(
    "How much did the author raise in seed funding from Idelle's husband"
    " (Julian) for Viaweb?",
)

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
resp_nodes = [n.node for n in init_response.source_nodes]

summary_index = SummaryIndex(resp_nodes)
query_engine = summary_index.as_query_engine(
    node_postprocessors=[node_postprocessor]
)
response = query_engine.query(query_str)

