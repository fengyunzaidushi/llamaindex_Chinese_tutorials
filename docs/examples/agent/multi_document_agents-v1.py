#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/agent/multi_document_agents-v1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Multi-Document Agents (V1)
# 

# 
# This is an extension of V0 multi-document agents with the additional features:
# - Reranking during document (tool) retrieval
# - Query planning tool that the agent can use to plan 
# 
# 
# We do this with the following architecture:
# 
# - setup a "document agent" over each Document: each doc agent can do QA/summarization within its doc
# - setup a top-level agent over this set of document agents. Do tool retrieval and then do CoT over the set of tools to answer a question.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# ## Setup and Download Data
# 

domain = "docs.llamaindex.ai"
docs_url = "https://docs.llamaindex.ai/en/latest/"
#('wget -e robots=off --recursive --no-clobber --page-requisites --html-extension --convert-links --restrict-file-names=windows --domains {domain} --no-parent {docs_url}')

from llama_hub.file.unstructured.base import UnstructuredReader
from pathlib import Path
from llama_index.llms import OpenAI
from llama_index import ServiceContext

reader = UnstructuredReader()

all_files_gen = Path("./docs.llamaindex.ai/").rglob("*")
all_files = [f.resolve() for f in all_files_gen]

all_html_files = [f for f in all_files if f.suffix.lower() == ".html"]

len(all_html_files)

from llama_index import Document

# TODO: set to higher value if you want more docs
doc_limit = 100

docs = []
for idx, f in enumerate(all_html_files):
    if idx > doc_limit:
        break
    print(f"Idx {idx}/{len(all_html_files)}")
    loaded_docs = reader.load_data(file=f, split_documents=True)
    # Hardcoded Index. Everything before this is ToC for all pages
    start_idx = 72
    loaded_doc = Document(
        text="\n\n".join([d.get_content() for d in loaded_docs[72:]]),
        metadata={"path": str(f)},
    )
    print(loaded_doc.metadata["path"])
    docs.append(loaded_doc)

# Define LLM + Service Context + Callback Manager

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

# ## Building Multi-Document Agents
# 

from llama_index import VectorStoreIndex, SummaryIndex

import nest_asyncio

nest_asyncio.apply()

# ### Build Document Agent for each Document
# 

# 
# We define both a vector index (for semantic search) and summary index (for summarization) for each document. The two query engines are then converted into tools that are passed to an OpenAI function calling agent.
# 
# This document agent can dynamically choose to perform semantic search or summarization within a given document.
# 
# We create a separate document agent for each city.

from llama_index.agent import OpenAIAgent
from llama_index import load_index_from_storage, StorageContext
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.node_parser import SentenceSplitter
import os
from tqdm.notebook import tqdm
import pickle

async def build_agent_per_doc(nodes, file_base):
    print(file_base)

    vi_out_path = f"./data/llamaindex_docs/{file_base}"
    summary_out_path = f"./data/llamaindex_docs/{file_base}_summary.pkl"
    if not os.path.exists(vi_out_path):
        Path("./data/llamaindex_docs/").mkdir(parents=True, exist_ok=True)
        # build vector index
        vector_index = VectorStoreIndex(nodes, service_context=service_context)
        vector_index.storage_context.persist(persist_dir=vi_out_path)
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vi_out_path),
            service_context=service_context,
        )

    # build summary index
    summary_index = SummaryIndex(nodes, service_context=service_context)

    # define query engines
    vector_query_engine = vector_index.as_query_engine()
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize"
    )

    # extract a summary
    if not os.path.exists(summary_out_path):
        Path(summary_out_path).parent.mkdir(parents=True, exist_ok=True)
        summary = str(
            await summary_query_engine.aquery(
                "Extract a concise 1-2 line summary of this document"
            )
        )
        pickle.dump(summary, open(summary_out_path, "wb"))
    else:
        summary = pickle.load(open(summary_out_path, "rb"))

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{file_base}",
                description=f"Useful for questions related to specific facts",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name=f"summary_tool_{file_base}",
                description=f"Useful for summarization questions",
            ),
        ),
    ]

    # build agent
    function_llm = OpenAI(model="gpt-4")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt=f"""\
You are a specialized agent designed to answer queries about the `{file_base}.html` part of the LlamaIndex docs.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
    )

    return agent, summary

async def build_agents(docs):
    node_parser = SentenceSplitter()

    # Build agents dictionary
    agents_dict = {}
    extra_info_dict = {}

    # # this is for the baseline
    # all_nodes = []

    for idx, doc in enumerate(tqdm(docs)):
        nodes = node_parser.get_nodes_from_documents([doc])
        # all_nodes.extend(nodes)

        # ID will be base + parent
        file_path = Path(doc.metadata["path"])
        file_base = str(file_path.parent.stem) + "_" + str(file_path.stem)
        agent, summary = await build_agent_per_doc(nodes, file_base)

        agents_dict[file_base] = agent
        extra_info_dict[file_base] = {"summary": summary, "nodes": nodes}

    return agents_dict, extra_info_dict

agents_dict, extra_info_dict = await build_agents(docs)

# ### Build Retriever-Enabled OpenAI Agent
# 
# We build a top-level agent that can orchestrate across the different document agents to answer any user query.
# 
# This `RetrieverOpenAIAgent` performs tool retrieval before tool use (unlike a default agent that tries to put all tools in the prompt).
# 
# **Improvements from V0**: We make the following improvements compared to the "base" version in V0.
# 
# - Adding in reranking: we use Cohere reranker to better filter the candidate set of documents.
# - Adding in a query planning tool: we add an explicit query planning tool that's dynamically created based on the set of retrieved tools.
# 

# define tool for each document agent
all_tools = []
for file_base, agent in agents_dict.items():
    summary = extra_info_dict[file_base]["summary"]
    doc_tool = QueryEngineTool(
        query_engine=agent,
        metadata=ToolMetadata(
            name=f"tool_{file_base}",
            description=summary,
        ),
    )
    all_tools.append(doc_tool)

print(all_tools[0].metadata)

# define an "object" index and retriever over these tools
from llama_index import VectorStoreIndex
from llama_index.objects import (
    ObjectIndex,
    SimpleToolNodeMapping,
    ObjectRetriever,
)
from llama_index.retrievers import BaseRetriever
from llama_index.postprocessor import CohereRerank
from llama_index.tools import QueryPlanTool
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.llms import OpenAI

llm = OpenAI(model_name="gpt-4-0613")

tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
obj_index = ObjectIndex.from_objects(
    all_tools,
    tool_mapping,
    VectorStoreIndex,
)
vector_node_retriever = obj_index.as_node_retriever(similarity_top_k=10)

# define a custom retriever with reranking
class CustomRetriever(BaseRetriever):
    def __init__(self, vector_retriever, postprocessor=None):
        self._vector_retriever = vector_retriever
        self._postprocessor = postprocessor or CohereRerank(top_n=5)
        super().__init__()

    def _retrieve(self, query_bundle):
        retrieved_nodes = self._vector_retriever.retrieve(query_bundle)
        filtered_nodes = self._postprocessor.postprocess_nodes(
            retrieved_nodes, query_bundle=query_bundle
        )

        return filtered_nodes

# define a custom object retriever that adds in a query planning tool
class CustomObjectRetriever(ObjectRetriever):
    def __init__(self, retriever, object_node_mapping, all_tools, llm=None):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping
        self._llm = llm or OpenAI("gpt-4-0613")

    def retrieve(self, query_bundle):
        nodes = self._retriever.retrieve(query_bundle)
        tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

        sub_question_sc = ServiceContext.from_defaults(llm=self._llm)
        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=tools, service_context=sub_question_sc
        )
        sub_question_description = f"""\
Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this \
tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
"""
        sub_question_tool = QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="compare_tool", description=sub_question_description
            ),
        )

        return tools + [sub_question_tool]

custom_node_retriever = CustomRetriever(vector_node_retriever)

# wrap it with ObjectRetriever to return objects
custom_obj_retriever = CustomObjectRetriever(
    custom_node_retriever, tool_mapping, all_tools, llm=llm
)

tmps = custom_obj_retriever.retrieve("hello")
print(len(tmps))

from llama_index.agent import FnRetrieverOpenAIAgent, ReActAgent

top_agent = FnRetrieverOpenAIAgent.from_retriever(
    custom_obj_retriever,
    system_prompt=""" \
You are an agent designed to answer queries about the documentation.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    llm=llm,
    verbose=True,
)

# top_agent = ReActAgent.from_tools(
#     tool_retriever=custom_obj_retriever,
#     system_prompt=""" \
# You are an agent designed to answer queries about the documentation.
# Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

# """,
#     llm=llm,
#     verbose=True,
# )

# ### Define Baseline Vector Store Index
# 
# As a point of comparison, we define a "naive" RAG pipeline which dumps all docs into a single vector index collection.
# 
# We set the top_k = 4

all_nodes = [
    n for extra_info in extra_info_dict.values() for n in extra_info["nodes"]
]

base_index = VectorStoreIndex(all_nodes)
base_query_engine = base_index.as_query_engine(similarity_top_k=4)

# ## Running Example Queries
# 
# Let's run some example queries, ranging from QA / summaries over a single document to QA / summarization over multiple documents.

response = top_agent.query(
    "Tell me about the different types of evaluation in LlamaIndex"
)

print(response)

# baseline
response = base_query_engine.query(
    "Tell me about the different types of evaluation in LlamaIndex"
)
print(str(response))

response = top_agent.query(
    "Compare the content in the contributions page vs. index page."
)

print(response)

response = top_agent.query(
    "Can you compare the tree index and list index at a very high-level?"
)

print(str(response))

