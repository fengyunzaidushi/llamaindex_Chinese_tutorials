#!/usr/bin/env python
# coding: utf-8

# # Recursive Retriever + Query Engine Demo 
# 

# 
# The concept of recursive retrieval is that we not only explore the directly most relevant nodes, but also explore
# node relationships to additional retrievers/query engines and execute them. For instance, a node may represent a concise summary of a structured table,
# and link to a SQL/Pandas query engine over that structured table. Then if the node is retrieved, we want to also query the underlying query engine for the answer.
# 
# This can be especially useful for documents with hierarchical relationships. In this example, we walk through a Wikipedia article about billionaires (in PDF form), which contains both text and a variety of embedded structured tables. We first create a Pandas query engine over each table, but also represent each table by an `IndexNode` (stores a link to the query engine); this Node is stored along with other Nodes in a vector store. 
# 
# During query-time, if an `IndexNode` is fetched, then the underlying query engine/retriever will be queried. 
# 
# **Notes about Setup**
# 
# We use `camelot` to extract text-based tables from PDFs.

import camelot
from llama_index import Document, SummaryIndex

# https://en.wikipedia.org/wiki/The_World%27s_Billionaires
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.query_engine import PandasQueryEngine, RetrieverQueryEngine
from llama_index.retrievers import RecursiveRetriever
from llama_index.schema import IndexNode
from llama_index.llms import OpenAI

from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from pathlib import Path
from typing import List

# ## Load in Document (and Tables)
# 
# We use our `PyMuPDFReader` to read in the main text of the document.
# 
# We also use `camelot` to extract some structured tables from the document

file_path = "billionaires_page.pdf"

# initialize PDF reader
reader = PyMuPDFReader()

docs = reader.load(file_path)

# use camelot to parse tables
def get_tables(path: str, pages: List[int]):
    table_dfs = []
    for page in pages:
        table_list = camelot.read_pdf(path, pages=str(page))
        table_df = table_list[0].df
        table_df = (
            table_df.rename(columns=table_df.iloc[0])
            .drop(table_df.index[0])
            .reset_index(drop=True)
        )
        table_dfs.append(table_df)
    return table_dfs

table_dfs = get_tables(file_path, pages=[3, 25])

# shows list of top billionaires in 2023
table_dfs[0]

# shows list of top billionaires
table_dfs[1]

# ## Create Pandas Query Engines
# 
# We create a pandas query engine over each structured table.
# 
# These can be executed on their own to answer queries about each table.

# define query engines over these tables
llm = OpenAI(model="gpt-4")

service_context = ServiceContext.from_defaults(llm=llm)
df_query_engines = [
    PandasQueryEngine(table_df, service_context=service_context)
    for table_df in table_dfs
]

response = df_query_engines[0].query(
    "What's the net worth of the second richest billionaire in 2023?"
)
print(str(response))

response = df_query_engines[1].query(
    "How many billionaires were there in 2009?"
)
print(str(response))

# ## Build Vector Index
# 
# Build vector index over the chunked document as well as over the additional `IndexNode` objects linked to the tables.

llm = OpenAI(temperature=0, model="gpt-4")

service_context = ServiceContext.from_defaults(
    llm=llm,
)

doc_nodes = service_context.node_parser.get_nodes_from_documents(docs)

# define index nodes
summaries = [
    (
        "This node provides information about the world's richest billionaires"
        " in 2023"
    ),
    (
        "This node provides information on the number of billionaires and"
        " their combined net worth from 2000 to 2023."
    ),
]

df_nodes = [
    IndexNode(text=summary, index_id=f"pandas{idx}")
    for idx, summary in enumerate(summaries)
]

df_id_query_engine_mapping = {
    f"pandas{idx}": df_query_engine
    for idx, df_query_engine in enumerate(df_query_engines)
}

# construct top-level vector index + query engine
vector_index = VectorStoreIndex(doc_nodes + df_nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# ## Use `RecursiveRetriever` in our `RetrieverQueryEngine`
# 
# We define a `RecursiveRetriever` object to recursively retrieve/query nodes. We then put this in our `RetrieverQueryEngine` along with a `ResponseSynthesizer` to synthesize a response.
# 
# We pass in mappings from id to retriever and id to query engine. We then pass in a root id representing the retriever we query first.

# baseline vector index (that doesn't include the extra df nodes).
# used to benchmark
vector_index0 = VectorStoreIndex(doc_nodes)
vector_query_engine0 = vector_index0.as_query_engine()

from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer

recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_id_query_engine_mapping,
    verbose=True,
)

response_synthesizer = get_response_synthesizer(
    # service_context=service_context,
    response_mode="compact"
)

query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever, response_synthesizer=response_synthesizer
)

response = query_engine.query(
    "What's the net worth of the second richest billionaire in 2023?"
)

response.source_nodes[0].node.get_content()

str(response)

response = query_engine.query("How many billionaires were there in 2009?")

str(response)

response = vector_query_engine0.query(
    "How many billionaires were there in 2009?"
)

print(response.source_nodes[0].node.get_content())

print(str(response))

response.source_nodes[0].node.get_content()

response = query_engine.query(
    "Which billionaires are excluded from this list?"
)

print(str(response))

