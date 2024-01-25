#!/usr/bin/env python
# coding: utf-8

# # Async Ingestion Pipeline + Metadata Extraction
# 
# Recently, LlamaIndex has introduced async metadata extraction. Let's compare metadata extraction speeds in an ingestion pipeline using a newer and older version of LlamaIndex.
# 
# We will test a pipeline using the classic Paul Graham essay.

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

import os

os.environ[
    "OPENAI_API_KEY"
] = "sk-...

# ## New LlamaIndex Ingestion
# 
# Using a version of LlamaIndex greater or equal to v0.9.7, we can take advantage of improved async metadata extraction within ingestion pipelines.
# 
# **NOTE:** Restart your notebook after installing a new version!

#('pip install "llama_index>=0.9.7"')

# **NOTE:** The `num_workers` kwarg controls how many requests can be outgoing at a given time using an async semaphore. Setting it higher may increase speeds, but can also lead to timeouts or rate limits, so set it wisely.

from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.ingestion import IngestionPipeline
from llama_index.extractors import TitleExtractor, SummaryExtractor
from llama_index.text_splitter import SentenceSplitter
from llama_index.schema import MetadataMode

def build_pipeline():
    llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1)

    transformations = [
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
        TitleExtractor(
            llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8
        ),
        SummaryExtractor(
            llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8
        ),
        OpenAIEmbedding(),
    ]

    return IngestionPipeline(transformations=transformations)

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham").load_data()

import time

times = []
for _ in range(3):
    time.sleep(30)  # help prevent rate-limits/timeouts, keeps each run fair
    pipline = build_pipeline()
    start = time.time()
    nodes = await pipline.arun(documents=documents)
    end = time.time()
    times.append(end - start)

print(f"Average time: {sum(times) / len(times)}")

# The current `openai` python client package is a tad unstable -- sometimes async jobs will timeout, skewing the average. You can see the last progress bar took 1 minute instead of the previous 6 or 7 seconds, skewing the average.

# ## Old LlamaIndex Ingestion
# 
# Now, lets compare to an older version of LlamaIndex, which was using "fake" async for metadata extraction.
# 
# **NOTE:** Restart your notebook after installing the new version!

#('pip install "llama_index<0.9.6"')

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.ingestion import IngestionPipeline
from llama_index.extractors import TitleExtractor, SummaryExtractor
from llama_index.text_splitter import SentenceSplitter
from llama_index.schema import MetadataMode

def build_pipeline():
    llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1)

    transformations = [
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
        TitleExtractor(llm=llm, metadata_mode=MetadataMode.EMBED),
        SummaryExtractor(llm=llm, metadata_mode=MetadataMode.EMBED),
        OpenAIEmbedding(),
    ]

    return IngestionPipeline(transformations=transformations)

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham").load_data()

import time

times = []
for _ in range(3):
    time.sleep(30)  # help prevent rate-limits/timeouts, keeps each run fair
    pipline = build_pipeline()
    start = time.time()
    nodes = await pipline.arun(documents=documents)
    end = time.time()
    times.append(end - start)

print(f"Average time: {sum(times) / len(times)}")

