#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/composable_indices/financial_data_analysis/DeepLakeDemo-FinancialData.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # DeepLake + LlamaIndex
# 
# Look at financial statements

#('pip install llama-index deeplake')

# My OpenAI Key
import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI token: ")

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
    download_loader,
    Document,
)
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.llms import OpenAI
from typing import List, Optional, Tuple
from pathlib import Path
import requests
import tqdm

# ##

# financial reports of amamzon, but can be replaced by any URLs of pdfs
urls = [
    "https://s2.q4cdn.com/299287126/files/doc_financials/Q1_2018_-_8-K_Press_Release_FILED.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/Q2_2018_Earnings_Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_news/archive/Q318-Amazon-Earnings-Press-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_news/archive/AMAZON.COM-ANNOUNCES-FOURTH-QUARTER-SALES-UP-20-TO-$72.4-BILLION.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/Q119_Amazon_Earnings_Press_Release_FINAL.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_news/archive/Amazon-Q2-2019-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_news/archive/Q3-2019-Amazon-Financial-Results.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_news/archive/Amazon-Q4-2019-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2020/Q1/AMZN-Q1-2020-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2020/q2/Q2-2020-Amazon-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2020/q4/Amazon-Q4-2020-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2021/q1/Amazon-Q1-2021-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2021/q2/AMZN-Q2-2021-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2021/q3/Q3-2021-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2021/q4/business_and_financial_update.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2022/q1/Q1-2022-Amazon-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2022/q2/Q2-2022-Amazon-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2022/q3/Q3-2022-Amazon-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2022/q4/Q4-2022-Amazon-Earnings-Release.pdf",
]

# hardcoding for now since we're missing q3 2020
years = [
    2018,
    2018,
    2018,
    2018,
    2019,
    2019,
    2019,
    2019,
    2020,
    2020,
    2020,
    2021,
    2021,
    2021,
    2021,
    2022,
    2022,
    2022,
    2022,
]
months = [1, 4, 7, 10, 1, 4, 7, 10, 1, 4, 10, 1, 4, 7, 10, 1, 4, 7, 10]

zipped_data = list(zip(urls, months, years))

PDFReader = download_loader("PDFReader")

loader = PDFReader()

def download_reports(
    data: List[Tuple[str, int, int]], out_dir: Optional[str] = None
) -> List[Document]:
    """Download pages from a list of urls."""
    docs = []
    out_dir = Path(out_dir or ".")
    if not out_dir.exists():
        print(out_dir)
        os.makedirs(out_dir)

    for url, month, year in tqdm.tqdm(data):
        path_base = url.split("/")[-1]
        out_path = out_dir / path_base
        if not out_path.exists():
            r = requests.get(url)
            with open(out_path, "wb") as f:
                f.write(r.content)
        doc = loader.load_data(file=Path(out_path))[0]

        date_str = f"{month:02d}" + "-01-" + str(year)
        doc.extra_info = {"Date": date_str}

        docs.append(doc)
    return docs

def _get_quarter_from_month(month: int) -> str:
    mapping = {1: "Q1", 4: "Q2", 7: "Q3", 10: "Q4"}
    return mapping[month]

docs = download_reports(zipped_data, "data")

# ### Build Vector Indices

llm_chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

service_context = ServiceContext.from_defaults(llm=llm_chatgpt)

# Build city document index
from llama_index.storage.storage_context import StorageContext

# build vector index for each quarterly statement, store in dictionary
dataset_root = "amazon_example/amazon_financial_"
vector_indices = {}
for idx, (_, month, year) in enumerate(zipped_data):
    doc = docs[idx]

    dataset_path = dataset_root + f"{month:02d}_{year}"
    vector_store = DeepLakeVectorStore(
        dataset_path=dataset_path,
        overwrite=True,
        verbose=False,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_index = VectorStoreIndex.from_documents(
        [doc], storage_context=storage_context, service_context=service_context
    )
    vector_indices[(month, year)] = vector_index

# #### Test Querying a Vector Index

response = (
    vector_indices[(1, 2018)]
    .as_query_engine(service_context=service_context)
    .query("What is the operating cash flow?")
)

print(str(response))
print(response.get_formatted_sources())

response = (
    vector_indices[(1, 2018)]
    .as_query_engine(service_context=service_context)
    .query("What are the updates on Whole Foods?")
)

print(response)

# ### Build Graph: Keyword Table Index on top of vector indices! 
# 
# We compose a keyword table index on top of all the vector indices.

from llama_index.indices.composability.graph import ComposableGraph

# set summary text for city
index_summaries = {}
for idx, (_, month, year) in enumerate(zipped_data):
    quarter_str = _get_quarter_from_month(month)
    index_summaries[
        (month, year)
    ] = f"Amazon Financial Statement, {quarter_str}, {year}"

graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [index for _, index in vector_indices.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50,
)

from llama_index.indices.query.query_transform.base import (
    DecomposeQueryTransform,
)

decompose_transform = DecomposeQueryTransform(
    service_context.llm, verbose=True
)

# TMP
query_str = "Analyze revenue in Q1 of 2018."

# with query decomposition in subindices
from llama_index.query_engine.transform_query_engine import (
    TransformQueryEngine,
)

custom_query_engines = {}
for index in vector_indices.values():
    query_engine = index.as_query_engine(service_context=service_context)
    transform_metadata = {"index_summary": index.index_struct.summary}
    tranformed_query_engine = TransformQueryEngine(
        query_engine,
        decompose_transform,
        transform_metadata=transform_metadata,
    )
    custom_query_engines[index.index_id] = tranformed_query_engine

custom_query_engines[
    graph.root_index.index_id
] = graph.root_index.as_query_engine(
    retriever_mode="simple",
    response_mode="tree_summarize",
    service_context=service_context,
)

query_engine_decompose = graph.as_query_engine(
    custom_query_engines=custom_query_engines,
)

from llama_index.indices.query.query_transform.base import (
    DecomposeQueryTransform,
)

decompose_transform = DecomposeQueryTransform(
    service_context.llm, verbose=True
)

response_chatgpt = query_engine_decompose.query(
    "Analyze revenue in Q1 of 2018."
)

print(str(response_chatgpt))

response_chatgpt = query_engine_decompose.query(
    "Analyze revenue in Q2 of 2018."
)

print(str(response_chatgpt))

response_chatgpt = query_engine_decompose.query(
    "Analyze and comapre revenue in Q1 and Q2 of 2018."
)

print(str(response_chatgpt))

