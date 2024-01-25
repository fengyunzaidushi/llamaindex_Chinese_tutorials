#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/SimpleIndexOnS3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # S3/R2 Storage

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
)
from IPython.#display import Markdown, #display

import dotenv
import s3fs
import os

dotenv.load_dotenv("../../../.env")

AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]
R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]

assert AWS_KEY is not None and AWS_KEY != ""

s3 = s3fs.S3FileSystem(
    key=AWS_KEY,
    secret=AWS_SECRET,
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    s3_additional_kwargs={"ACL": "public-read"},
)

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
print(len(documents))

index = VectorStoreIndex.from_documents(documents, fs=s3)

# save index to disk
index.set_index_id("vector_index")
index.storage_context.persist("llama-index/storage_demo", fs=s3)

s3.listdir("llama-index/storage_demo")

# load index from s3
sc = StorageContext.from_defaults(
    persist_dir="llama-index/storage_demo", fs=s3
)

index2 = load_index_from_storage(sc, "vector_index")

index2.docstore.docs.keys()

