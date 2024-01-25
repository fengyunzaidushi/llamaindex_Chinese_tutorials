#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/callbacks/HoneyHiveLlamaIndexTracer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # HoneyHive LlamaIndex Tracer
# 
# [HoneyHive](https://honeyhive.ai) is a platform that helps developers monitor, evaluate and continuously improve their LLM-powered applications.
# 
# The `HoneyHiveLlamaIndexTracer` is integrated with HoneyHive to help developers debug and analyze the execution flow of your LLM pipeline, or to let developers customize feedback on specific trace events to create evaluation or fine-tuning datasets from production.
# 

import os
from getpass import getpass

if os.getenv("OPENAI_API_KEY") is None:
    os.environ["OPENAI_API_KEY"] = getpass(
        "Paste your OpenAI key from:"
        " https://platform.openai.com/account/api-keys\n"
    )
assert os.getenv("OPENAI_API_KEY", "").startswith(
    "sk-"
), "This doesn't look like a valid OpenAI API key"
print("OpenAI API key configured")

import os
from getpass import getpass

if os.getenv("HONEYHIVE_API_KEY") is None:
    os.environ["HONEYHIVE_API_KEY"] = getpass(
        "Paste your HoneyHive key from:"
        " https://app.honeyhive.ai/settings/account\n"
    )
print("HoneyHive API key configured")

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.callbacks import CallbackManager
from llama_index.callbacks import LlamaDebugHandler
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    SimpleDirectoryReader,
    SimpleKeywordTableIndex,
    StorageContext,
)
from llama_index.indices.composability import ComposableGraph
from llama_index import load_index_from_storage, load_graph_from_storage
from llama_index.llms import OpenAI
from honeyhive.utils.llamaindex_tracer import HoneyHiveLlamaIndexTracer

# ## Setup LLM

llm = OpenAI(model="gpt-4", temperature=0)

# ## HoneyHive Callback Manager Setup

# **Option 1**: Set Global Evaluation Handler

from llama_index import set_global_handler

set_global_handler(
    "honeyhive",
    project="My LlamaIndex Project",
    name="My LlamaIndex Pipeline",
    api_key=os.environ["HONEYHIVE_API_KEY"],
)
hh_tracer = llama_index.global_handler

service_context = ServiceContext.from_defaults(llm=llm)

# **Option 2**: Manually Configure Callback Handler
# 
# Also configure a debugger handler for extra notebook visibility.

llama_debug = LlamaDebugHandler(print_trace_on_end=True)

hh_tracer = HoneyHiveLlamaIndexTracer(
    project="My LlamaIndex Project",
    name="My LlamaIndex Pipeline",
    api_key=os.environ["HONEYHIVE_API_KEY"],
)

callback_manager = CallbackManager([llama_debug, hh_tracer])

service_context = ServiceContext.from_defaults(
    callback_manager=callback_manager, llm=llm
)

# ## 1. Indexing

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

docs = SimpleDirectoryReader("./data/paul_graham/").load_data()

index = VectorStoreIndex.from_documents(docs, service_context=service_context)

# ## 2. Query Over Index

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response, sep="\n")

# ## 3. Build Complex Indices

# fetch "New York City" page from Wikipedia
from pathlib import Path

import requests

response = requests.get(
    "https://en.wikipedia.org/w/api.php",
    params={
        "action": "query",
        "format": "json",
        "titles": "New York City",
        "prop": "extracts",
        "explaintext": True,
    },
).json()
page = next(iter(response["query"]["pages"].values()))
nyc_text = page["extract"]

data_path = Path("data")
if not data_path.exists():
    Path.mkdir(data_path)

with open("data/nyc_text.txt", "w") as fp:
    fp.write(nyc_text)

# load NYC dataset
nyc_documents = SimpleDirectoryReader("data/").load_data()
# load PG's essay
essay_documents = SimpleDirectoryReader("../data/paul_graham").load_data()

# While building a composable index, to correctly save the index,
# the same `storage_context` needs to be passed to every index.
storage_context = StorageContext.from_defaults()

# build NYC index
nyc_index = VectorStoreIndex.from_documents(
    nyc_documents,
    service_context=service_context,
    storage_context=storage_context,
)

# build essay index
essay_index = VectorStoreIndex.from_documents(
    essay_documents,
    service_context=service_context,
    storage_context=storage_context,
)

# ### 3.1. Query Over Graph Index

nyc_index_summary = """
    New York, often called New York City or NYC, 
    is the most populous city in the United States. 
    With a 2020 population of 8,804,190 distributed over 300.46 square miles (778.2 km2), 
    New York City is also the most densely populated major city in the United States, 
    and is more than twice as populous as second-place Los Angeles. 
    New York City lies at the southern tip of New York State, and 
    constitutes the geographical and demographic center of both the 
    Northeast megalopolis and the New York metropolitan area, the 
    largest metropolitan area in the world by urban landmass.[8] With over 
    20.1 million people in its metropolitan statistical area and 23.5 million 
    in its combined statistical area as of 2020, New York is one of the world's 
    most populous megacities, and over 58 million people live within 250 mi (400 km) of 
    the city. New York City is a global cultural, financial, and media center with 
    a significant influence on commerce, health care and life sciences, entertainment, 
    research, technology, education, politics, tourism, dining, art, fashion, and sports. 
    Home to the headquarters of the United Nations, 
    New York is an important center for international diplomacy,
    an established safe haven for global investors, and is sometimes described as the capital of the world.
"""
essay_index_summary = """
    Author: Paul Graham. 
    The author grew up painting and writing essays. 
    He wrote a book on Lisp and did freelance Lisp hacking work to support himself. 
    He also became the de facto studio assistant for Idelle Weber, an early photorealist painter. 
    He eventually had the idea to start a company to put art galleries online, but the idea was unsuccessful. 
    He then had the idea to write software to build online stores, which became the basis for his successful company, Viaweb. 
    After Viaweb was acquired by Yahoo!, the author returned to painting and started writing essays online. 
    He wrote a book of essays, Hackers & Painters, and worked on spam filters. 
    He also bought a building in Cambridge to use as an office. 
    He then had the idea to start Y Combinator, an investment firm that would 
    make a larger number of smaller investments and help founders remain as CEO. 
    He and his partner Jessica Livingston ran Y Combinator and funded a batch of startups twice a year. 
    He also continued to write essays, cook for groups of friends, and explore the concept of invented vs discovered in software. 
"""

from llama_index import StorageContext, load_graph_from_storage

graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [nyc_index, essay_index],
    index_summaries=[nyc_index_summary, essay_index_summary],
    max_keywords_per_chunk=50,
    service_context=service_context,
    storage_context=storage_context,
)

# ### 3.2 Query

query_engine = graph.as_query_engine()

response = query_engine.query(
    "What is the climate of New York City like? How cold is it during the"
    " winter?",
)
print(response, sep="\n")

# ## View HoneyHive Traces
# 
# When we are done tracing our events we can view them via [the HoneyHive platform](https://app.honeyhive.ai). Simply login to HoneyHive, go to your `My LlamaIndex Project` project, click the `Data Store` tab and view your `Sessions`.
