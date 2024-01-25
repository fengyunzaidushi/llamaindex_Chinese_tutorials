#!/usr/bin/env python
# coding: utf-8

# # Structured Hierarchical Retrieval
# 
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# Doing RAG well over multiple documents is hard. A general framework is given a user query, first select the relevant documents before selecting the content inside.
# 
# But selecting the documents can be tough - how can we dynamically select documents based on different properties depending on the user query? 
# 

# 
# - Represent each document as a concise **metadata** dictionary containing different properties: an extracted summary along with structured metadata.
# - Store this metadata dictionary as filters within a vector database.
# - Given a user query, first do **auto-retrieval** - infer the relevant semantic query and the set of filters to query this data (effectively combining text-to-SQL and semantic search).

#('pip install llama-index')

# ## Setup and Download Data
# 

import nest_asyncio

nest_asyncio.apply()

import os

os.environ["GITHUB_TOKEN"] = ""

import os

from llama_hub.github_repo_issues import (
    GitHubRepositoryIssuesReader,
    GitHubIssuesClient,
)

github_client = GitHubIssuesClient()
loader = GitHubRepositoryIssuesReader(
    github_client,
    owner="run-llama",
    repo="llama_index",
    verbose=True,
)

orig_docs = loader.load_data()

limit = 100

docs = []
for idx, doc in enumerate(orig_docs):
    doc.metadata["index_id"] = doc.id_
    if idx >= limit:
        break
    docs.append(doc)

from copy import deepcopy
import asyncio
from tqdm.asyncio import tqdm_asyncio
from llama_index import SummaryIndex, Document, ServiceContext
from llama_index.llms import OpenAI
from llama_index.async_utils import run_jobs

async def aprocess_doc(doc, include_summary: bool = True):
    """Process doc."""
    print(f"Processing {doc.id_}")
    metadata = doc.metadata

    date_tokens = metadata["created_at"].split("T")[0].split("-")
    year = int(date_tokens[0])
    month = int(date_tokens[1])
    day = int(date_tokens[2])

    assignee = (
        "" if "assignee" not in doc.metadata else doc.metadata["assignee"]
    )
    size = ""
    if len(doc.metadata["labels"]) > 0:
        size_arr = [l for l in doc.metadata["labels"] if "size:" in l]
        size = size_arr[0].split(":")[1] if len(size_arr) > 0 else ""
    new_metadata = {
        "state": metadata["state"],
        "year": year,
        "month": month,
        "day": day,
        "assignee": assignee,
        "size": size,
        "index_id": doc.id_,
    }

    # now extract out summary
    summary_index = SummaryIndex.from_documents([doc])
    query_str = "Give a one-sentence concise summary of this issue."
    query_engine = summary_index.as_query_engine(
        service_context=ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo")
        )
    )
    summary_txt = str(query_engine.query(query_str))

    new_doc = Document(text=summary_txt, metadata=new_metadata)
    return new_doc

async def aprocess_docs(docs):
    """Process metadata on docs."""

    new_docs = []
    tasks = []
    for doc in docs:
        task = aprocess_doc(doc)
        tasks.append(task)

    new_docs = await run_jobs(tasks, show_progress=True, workers=5)

    # new_docs = await tqdm_asyncio.gather(*tasks)

    return new_docs

new_docs = await aprocess_docs(docs)

new_docs[5].metadata

# ## Load Data into Vector Store
# 
# We load both the summarized metadata as well as the original docs into the vector database.
# 1. **Summarized Metadata**: This goes into the `LlamaIndex_auto` collection.
# 2. **Original Docs**: This goes into the `LlamaIndex_AutoDoc` collection.
# 
# By storing both the summarized metadata as well as the original documents, we can execute our structured, hierarchical retrieval strategies.
# 
# We load into a vector database that supports auto-retrieval. 

# ### Load Summarized Metadata
# 
# This goes into `LlamaIndex_auto`

from llama_index.vector_stores import WeaviateVectorStore
from llama_index.storage import StorageContext
from llama_index import VectorStoreIndex

import weaviate

# cloud
auth_config = weaviate.AuthApiKey(api_key="")
client = weaviate.Client(
    "https://<weaviate-cluster>.weaviate.network",
    auth_client_secret=auth_config,
)

class_name = "LlamaIndex_auto"

# optional: delete schema
client.schema.delete_class(class_name)

vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name=class_name
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Since "new_docs" are concise summaries, we can directly feed them as nodes into VectorStoreIndex
index = VectorStoreIndex(new_docs, storage_context=storage_context)

# ### Load Original Docs
# 
# This goes into `LlamaIndex_AutoDoc`. 
# 

docs[0].metadata

# optional: delete schema
doc_class_name = "LlamaIndex_AutoDoc"
client.schema.delete_class(doc_class_name)

# construct separate Weaviate Index with original docs. Define a separate query engine with query engine mapping to each doc id.
vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name=doc_class_name
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

doc_index = VectorStoreIndex.from_documents(
    docs, storage_context=storage_context
)

# ## Setup Auto-Retriever
# 

# 
# 1. **Define the Schema**: Define the vector db schema (e.g. the metadata fields). This will be put into the LLM input prompt when it's deciding what metadata filters to infer.
# 2. **Instantiate the VectorIndexAutoRetriever class**: This creates a retriever on top of our summarized metadata index, and takes in the defined schema as input.
# 3. **Define a wrapper retriever**: This allows us to postprocess each node into an `IndexNode`, with an index id linking back source document. This will allow us to do recursive retrieval in the next section (which depends on IndexNode objects linking to downstream retrievers/query engines/other Nodes). **NOTE**: We are working on improving this abstraction.

# ### 1. Define the Schema

from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo

vector_store_info = VectorStoreInfo(
    content_info="Github Issues",
    metadata_info=[
        MetadataInfo(
            name="state",
            description="Whether the issue is `open` or `closed`",
            type="string",
        ),
        MetadataInfo(
            name="year",
            description="The year issue was created",
            type="integer",
        ),
        MetadataInfo(
            name="month",
            description="The month issue was created",
            type="integer",
        ),
        MetadataInfo(
            name="day",
            description="The day issue was created",
            type="integer",
        ),
        MetadataInfo(
            name="assignee",
            description="The assignee of the ticket",
            type="string",
        ),
        MetadataInfo(
            name="size",
            description="How big the issue is (XS, S, M, L, XL, XXL)",
            type="string",
        ),
    ],
)

# ### 2. Instantiate VectorIndexAutoRetriever

from llama_index.retrievers import VectorIndexAutoRetriever

retriever = VectorIndexAutoRetriever(
    index,
    vector_store_info=vector_store_info,
    similarity_top_k=2,
    empty_query_top_k=10,  # if only metadata filters are specified, this is the limit
    verbose=True,
)

# #### Try It Out
# 
# We can try out our autoretriever on its own.

nodes = retriever.retrieve("Tell me about some issues on 12/11")
print(f"Number retrieved: {len(nodes)}")
print(nodes[0].metadata)

# ### 3. Define a Wrapper Retriever

from llama_index.retrievers import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import IndexNode, NodeWithScore

class IndexAutoRetriever(BaseRetriever):
    """Index auto-retriever."""

    def __init__(self, retriever: VectorIndexAutoRetriever):
        """Init params."""
        self.retriever = retriever

    def _retrieve(self, query_bundle: QueryBundle):
        """Convert nodes to index node."""
        retrieved_nodes = self.retriever.retrieve(query_bundle)
        new_retrieved_nodes = []
        for retrieved_node in retrieved_nodes:
            index_id = retrieved_node.metadata["index_id"]
            index_node = IndexNode.from_text_node(
                retrieved_node.node, index_id=index_id
            )
            new_retrieved_nodes.append(
                NodeWithScore(node=index_node, score=retrieved_node.score)
            )
        return new_retrieved_nodes

index_retriever = IndexAutoRetriever(retriever=retriever)

# ## Setup Recursive Retriever
# 
# Now we setup a recursive retriever over our data. A recursive retriever links each node of one retriever to another retriever, query engine, or Node.
# 

# 
# We set this up through the following:
# 
# 1. **Define one retriever per document**: Put this in a dictionary
# 2. **Define our recursive retriever**: Add the root retriever (the summarized metadata retriever), and add the other document-specific retrievers in the arguments.

# ### 1. Define Per-Document Retriever

from llama_index.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

retriever_dict = {}
query_engine_dict = {}
for doc in docs:
    index_id = doc.metadata["index_id"]
    # filter for the specific doc id
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="index_id", operator=FilterOperator.EQ, value=index_id
            ),
        ]
    )
    retriever = doc_index.as_retriever(filters=filters)
    query_engine = doc_index.as_query_engine(filters=filters)

    retriever_dict[index_id] = retriever
    query_engine_dict[index_id] = query_engine

# ### 2. Define Recursive Retriever
# 
# We can now define our recursive retriever, which will first query the summaries and then retrieve the underlying docs.

from llama_index.retrievers import RecursiveRetriever

# note: can pass `agents` dict as `query_engine_dict` since every agent can be used as a query engine
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": index_retriever, **retriever_dict},
    # query_engine_dict=query_engine_dict,
    verbose=True,
)

# ## Try It Out
# 
# Now we can start retrieving relevant context over Github Issues! 
# 
# To complete the RAG pipeline setup we'll combine our recursive retriever with our `RetrieverQueryEngine` to generate a response in addition to the retrieved nodes.

# ### Try Out Retrieval

nodes = recursive_retriever.retrieve("Tell me about some issues on 12/11")

# If you ran the above, you should've gotten a long output in the logs. 
# 
# The result is the source chunks in the relevant docs. 
# 
# Let's look at the date attached to the source chunk (was present in the original metadata).

print(f"Number of source nodes: {len(nodes)}")
nodes[0].node.metadata

# ### Plug into `RetrieverQueryEngine`
# 
# We plug into RetrieverQueryEngine to synthesize a result.

from llama_index.query_engine import RetrieverQueryEngine
from llama_index.llms import OpenAI
from llama_index import ServiceContext

llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

query_engine = RetrieverQueryEngine.from_args(recursive_retriever, llm=llm)

response = query_engine.query("Tell me about some issues on 12/11")

print(str(response))

response = query_engine.query(
    "Tell me about some open issues related to agents"
)

print(str(response))

response = query_engine.query(
    "Tell me about some size S issues related to our llm integrations"
)

print(str(response))

# ## Concluding Thoughts
# 
# This shows you how to create a structured retrieval layer over your document summaries, allowing you to dynamically pull in the relevant documents based on the user query.
# 
# You may notice similarities between this and our [multi-document agents](https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents.html). Both architectures are aimed for powerful multi-document retrieval.
# 
# The goal of this notebook is to show you how to apply structured querying in a multi-document setting. You can actually apply this auto-retrieval algorithm to our multi-agent setup too. The multi-agent setup is primarily focused on adding agentic reasoning across documents and per documents, alloinwg multi-part queries using chain-of-thought.
