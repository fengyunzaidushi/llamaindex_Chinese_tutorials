#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/managed/vectaraDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Managed Index with Zilliz Cloud Pipeline
# 
# [Zilliz Cloud Pipelines](https://docs.zilliz.com/docs/pipelines) is a robust solution that efficiently transforms unstructured data into a vector database for effective semantic search.
# 
# ## Setup
# 
# 1. Install llama-index

# ! pip install llama-index

# 2. Set your [OpenAI](https://platform.openai.com) & [Zilliz Cloud](https://cloud.zilliz.com/) accounts

from getpass import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API Key:")

ZILLIZ_CLUSTER_ID = getpass("Enter your Zilliz Cluster ID:")
ZILLIZ_TOKEN = getpass("Enter your Zilliz Token:")

# #
# 
# ### From Signed URL
# 
# Zilliz Cloud Pipeline is able to ingest & automatically index a document given a presigned url.

from llama_index.indices import ZillizCloudPipelineIndex

zcp_index = ZillizCloudPipelineIndex.from_document_url(
    url="https://publicdataset.zillizcloud.com/milvus_doc.md",  # a public or pre-signed url of a file stored on s3 or gcs
    cluster_id=ZILLIZ_CLUSTER_ID,
    token=ZILLIZ_TOKEN,
    metadata={"version": "2.3"},
)

zcp_index.insert_doc_url(
    url="https://publicdataset.zillizcloud.com/milvus_doc_22.md",
    metadata={"version": "2.2"},
)

# ### From Local File
# 
# Coming soon.
# 
# ### From Raw Text
# 
# Coming soon.

# ## Working as Query Engine
# 
# A Zilliz Cloud Pipeline's Index can work as a Query Engine in LlamaIndex.
# It allows users to customize some parameters:
# - search_top_k: How many text nodes/chunks retrieved. Optional, defaults to `DEFAULT_SIMILARITY_TOP_K` (2).
# - filters: Metadata filters. Optional, defaults to None.
# - output_metadata: What metadata fields included in each retrieved text node. Optional, defaults to [].
# 
# It is optional to apply filters. For example, if we want to ask about Milvus 2.3, then we can set version as 2.3 in filters.

# # Get index without ingestion:
# from llama_index.indices import ZillizCloudPipelineIndex

# zcp_index = ZillizCloudPipelineIndex(
#         cluster_id=ZILLIZ_CLUSTER_ID,
#         token=ZILLIZ_TOKEN,
#         # collection_name='zcp_llamalection'
#     )

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

query_engine_milvus23 = zcp_index.as_query_engine(
    search_top_k=3,
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(key="version", value="2.3")
        ]  # version == "2.3"
    ),
    output_metadata=["version"],
)

# Then the query engine is ready for Semantic Search or Retrieval Augmented Generation with Milvus 2.3 documents:
# 
# - **Retrieve** (Semantic search powered by Zilliz Cloud Pipeline's Index):

question = "Can users delete entities by filtering non-primary fields?"
retrieved_nodes = query_engine_milvus23.retrieve(question)
print(retrieved_nodes)

# - **Query** (RAG powered by Zilliz Cloud Pipeline's Index & OpenAI's LLM):

response = query_engine_milvus23.query(question)
print(response.response)

