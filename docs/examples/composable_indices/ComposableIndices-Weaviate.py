#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/composable_indices/ComposableIndices-Weaviate.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Composable Graph with Weaviate

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys
import weaviate
from pprint import pprint

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SummaryIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.vector_stores import WeaviateVectorStore

resource_owner_config = weaviate.AuthClientPassword(
    username="<username>",
    password="<password>",
)

client = weaviate.Client(
    "https://test-weaviate-cluster.semi.network/",
    auth_client_secret=resource_owner_config,
)

# [optional] set batch
client.batch.configure(batch_size=10)

# #### Load Datasets
# 
# Load both the NYC Wikipedia page as well as Paul Graham's "What I Worked On" essay

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
        # 'exintro': True,
        "explaintext": True,
    },
).json()
page = next(iter(response["query"]["pages"].values()))
nyc_text = page["extract"]

data_path = Path("data/test_wiki")
if not data_path.exists():
    Path.mkdir(data_path)

with open("./data/test_wiki/nyc_text.txt", "w") as fp:
    fp.write(nyc_text)

# load NYC dataset
nyc_documents = SimpleDirectoryReader("./data/test_wiki").load_data()

# Download Paul Graham Essay data

#("mkdir -p 'data/paul_graham_essay/'")
#("wget 'https://github.com/jerryjliu/llama_index/blob/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham_essay/paul_graham_essay.txt'")

# load PG's essay
essay_documents = SimpleDirectoryReader("./data/paul_graham_essay").load_data()

# ### Building the document indices
# Build a tree index for the NYC wiki page and PG essay

# build NYC index
from llama_index.storage.storage_context import StorageContext

vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="Nyc_docs"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
nyc_index = VectorStoreIndex.from_documents(
    nyc_documents, storage_context=storage_context
)

# build essay index
vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="Essay_docs"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
essay_index = VectorStoreIndex.from_documents(
    essay_documents, storage_context=storage_context
)

# ### Set summaries for the indices
# 
# Add text summaries to indices, so we can compose other indices on top of it

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
index_summaries = [nyc_index_summary, essay_index_summary]
nyc_index.set_index_id("nyc_index")
essay_index.set_index_id("essay_index")

# ### Build Keyword Table Index on top of vector indices! 
# 
# We set summaries for each of the NYC and essay indices, and then compose a keyword index on top of it.

# ### Define Graph

from llama_index.indices.composability import ComposableGraph

graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [nyc_index, essay_index],
    index_summaries=index_summaries,
    max_keywords_per_chunk=50,
)

custom_query_engines = {
    graph.root_id: graph.root_index.as_query_engine(retriever_mode="simple")
}

query_engine = graph.as_query_engine(
    custom_query_engines=custom_query_engines,
)

# set Logging to DEBUG for more detailed outputs
# ask it a question about NYC
response = query_engine.query(
    "What is the weather of New York City like? How cold is it during the"
    " winter?",
)

print(str(response))

# Get source of response
print(response.get_formatted_sources())

# ask it a question about PG's essay
response = query_engine.query(
    "What did the author do growing up, before his time at Y Combinator?",
)

print(str(response))

# Get source of response
print(response.get_formatted_sources())

