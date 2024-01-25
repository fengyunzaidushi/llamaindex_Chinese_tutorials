#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/OpensearchDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Opensearch Vector Store
# 
# Elasticsearch only supports Lucene indices, so only Opensearch is supported.

# **Note on setup**: We setup a local Opensearch instance through the following doc. https://opensearch.org/docs/1.0/
# 
# If you run into SSL issues, try the following `docker run` command instead: 
# ```
# docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "plugins.security.disabled=true" opensearchproject/opensearch:1.0.1
# ```
# 
# Reference: https://github.com/opensearch-project/OpenSearch/issues/1598

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

from os import getenv
from llama_index import SimpleDirectoryReader
from llama_index.vector_stores import (
    OpensearchVectorStore,
    OpensearchVectorClient,
)
from llama_index import VectorStoreIndex, StorageContext

# http endpoint for your cluster (opensearch required for vector index usage)
endpoint = getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
# index to demonstrate the VectorStore impl
idx = getenv("OPENSEARCH_INDEX", "gpt-index-demo")
# load some sample data
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# OpensearchVectorClient stores text in this field by default
text_field = "content"
# OpensearchVectorClient stores embeddings in this field by default
embedding_field = "embedding"
# OpensearchVectorClient encapsulates logic for a
# single opensearch index with vector search enabled
client = OpensearchVectorClient(
    endpoint, idx, 1536, embedding_field=embedding_field, text_field=text_field
)
# initialize vector store
vector_store = OpensearchVectorStore(client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# initialize an index using our sample data and the client we just created
index = VectorStoreIndex.from_documents(
    documents=documents, storage_context=storage_context
)

# run query
query_engine = index.as_query_engine()
res = query_engine.query("What did the author do growing up?")
res.response

# The OpenSearch vector store supports [filter-context queries](https://opensearch.org/docs/latest/query-dsl/query-filter-context/).

from llama_index import Document
from llama_index.vector_stores.types import MetadataFilters, ExactMatchFilter
import regex as re

# Split the text into paragraphs.
text_chunks = documents[0].text.split("\n\n")

# Create a document for each footnote
footnotes = [
    Document(
        text=chunk,
        id=documents[0].doc_id,
        metadata={"is_footnote": bool(re.search(r"^\s*\[\d+\]\s*", chunk))},
    )
    for chunk in text_chunks
    if bool(re.search(r"^\s*\[\d+\]\s*", chunk))
]

for f in footnotes:
    index.insert(f)

# Create a query engine that only searches certain footnotes.
footnote_query_engine = index.as_query_engine(
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="term", value='{"metadata.is_footnote": "true"}'
            ),
            ExactMatchFilter(
                key="query_string",
                value='{"query": "content: space AND content: lisp"}',
            ),
        ]
    )
)

res = footnote_query_engine.query(
    "What did the author about space aliens and lisp?"
)
res.response

# ## Use reader to check out what VectorStoreIndex just created in our index.
# 
# Reader works with Elasticsearch too as it just uses the basic search features.

# create a reader to check out the index used in previous section.
from llama_index.readers import ElasticsearchReader

rdr = ElasticsearchReader(endpoint, idx)
# set embedding_field optionally to read embedding data from the elasticsearch index
docs = rdr.load_data(text_field, embedding_field=embedding_field)
# docs have embeddings in them
print("embedding dimension:", len(docs[0].embedding))
# full document is stored in metadata
print("all fields in index:", docs[0].metadata.keys())

# we can check out how the text was chunked by the `GPTOpensearchIndex`
print("total number of chunks created:", len(docs))

# search index using standard elasticsearch query DSL
docs = rdr.load_data(text_field, {"query": {"match": {text_field: "Lisp"}}})
print("chunks that mention Lisp:", len(docs))
docs = rdr.load_data(text_field, {"query": {"match": {text_field: "Yahoo"}}})
print("chunks that mention Yahoo:", len(docs))

# ## Hybrid query for opensearch vector store
# Hybrid query has been supported since OpenSearch 2.10. It is a combination of vector search and text search. It is useful when you want to search for a specific text and also want to filter the results by vector similarity. You can find more details: https://opensearch.org/docs/latest/query-dsl/compound/hybrid/. 

# ##

from os import getenv
from llama_index.vector_stores import (
    OpensearchVectorStore,
    OpensearchVectorClient,
)

# http endpoint for your cluster (opensearch required for vector index usage)
endpoint = getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
# index to demonstrate the VectorStore impl
idx = getenv("OPENSEARCH_INDEX", "auto_retriever_movies")

# OpensearchVectorClient stores text in this field by default
text_field = "content"
# OpensearchVectorClient stores embeddings in this field by default
embedding_field = "embedding"
# OpensearchVectorClient encapsulates logic for a
# single opensearch index with vector search enabled with hybrid search pipeline
client = OpensearchVectorClient(
    endpoint,
    idx,
    4096,
    embedding_field=embedding_field,
    text_field=text_field,
    search_pipeline="hybrid-search-pipeline",
)

from llama_index.embeddings import OllamaEmbedding

embed_model = OllamaEmbedding(model_name="llama2")

# initialize vector store
vector_store = OpensearchVectorStore(client)

# ### Prepare the index

from llama_index.schema import TextNode
from llama_index import VectorStoreIndex, StorageContext, ServiceContext

storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(
    embed_model=embed_model, llm=None
)

nodes = [
    TextNode(
        text="The Shawshank Redemption",
        metadata={
            "author": "Stephen King",
            "theme": "Friendship",
        },
    ),
    TextNode(
        text="The Godfather",
        metadata={
            "director": "Francis Ford Coppola",
            "theme": "Mafia",
        },
    ),
    TextNode(
        text="Inception",
        metadata={
            "director": "Christopher Nolan",
        },
    ),
]

index = VectorStoreIndex(
    nodes, storage_context=storage_context, service_context=service_context
)

# ### Search the index with hybrid query by specifying the vector store query mode: VectorStoreQueryMode.HYBRID with filters

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores.types import VectorStoreQueryMode

filters = MetadataFilters(
    filters=[
        ExactMatchFilter(
            key="term", value='{"metadata.theme.keyword": "Mafia"}'
        )
    ]
)

retriever = index.as_retriever(
    filters=filters, vector_store_query_mode=VectorStoreQueryMode.HYBRID
)

result = retriever.retrieve("What is inception about?")

print(result)

