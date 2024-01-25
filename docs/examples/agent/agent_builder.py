#!/usr/bin/env python
# coding: utf-8

# # GPT Builder Demo
# 
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/agent/agent_builder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 

# 
# Here you can build your own agent...with another agent!

from llama_index.tools import BaseTool, FunctionTool

from llama_index.agent import OpenAIAgent
from llama_index.prompts import PromptTemplate
from llama_index.llms import ChatMessage, OpenAI
from llama_index import ServiceContext

llm = OpenAI(model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)

# ## Define Candidate Tools
# 
# We also define a tool retriever to retrieve candidate tools.
# 

from llama_index import SimpleDirectoryReader

wiki_titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]

from pathlib import Path

import requests

for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

# Load all wiki documents
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()

# ### Build Query Tool for Each Document

from llama_index.agent import OpenAIAgent
from llama_index.tools import QueryEngineTool, ToolMetadata

# Build tool dictionary
tool_dict = {}

for wiki_title in wiki_titles:
    # build vector index
    vector_index = VectorStoreIndex.from_documents(
        city_docs[wiki_title], service_context=service_context
    )
    # define query engines
    vector_query_engine = vector_index.as_query_engine()

    # define tools
    vector_tool = QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name=wiki_title,
            description=("Useful for questions related to" f" {wiki_title}"),
        ),
    )
    tool_dict[wiki_title] = vector_tool

# ### Define Tool Retriever

# define an "object" index and retriever over these tools
from llama_index import VectorStoreIndex
from llama_index.objects import ObjectIndex, SimpleToolNodeMapping

tool_mapping = SimpleToolNodeMapping.from_objects(list(tool_dict.values()))
tool_index = ObjectIndex.from_objects(
    list(tool_dict.values()),
    tool_mapping,
    VectorStoreIndex,
)
tool_retriever = tool_index.as_retriever(similarity_top_k=1)

# ### Load Data
# 
# Here we load wikipedia pages from different cities.

# ## Define Meta-Tools for GPT Builder

from llama_index.prompts import ChatPromptTemplate
from typing import List

GEN_SYS_PROMPT_STR = """\
Task information is given below. 

Given the task, please generate a system prompt for an OpenAI-powered bot to solve this task: 
{task} \
"""

gen_sys_prompt_messages = [
    ChatMessage(
        role="system",
        content="You are helping to build a system prompt for another bot.",
    ),
    ChatMessage(role="user", content=GEN_SYS_PROMPT_STR),
]

GEN_SYS_PROMPT_TMPL = ChatPromptTemplate(gen_sys_prompt_messages)

agent_cache = {}

def create_system_prompt(task: str):
    """Create system prompt for another agent given an input task."""
    llm = OpenAI(llm="gpt-4")
    fmt_messages = GEN_SYS_PROMPT_TMPL.format_messages(task=task)
    response = llm.chat(fmt_messages)
    return response.message.content

def get_tools(task: str):
    """Get the set of relevant tools to use given an input task."""
    subset_tools = tool_retriever.retrieve(task)
    return [t.metadata.name for t in subset_tools]

def create_agent(system_prompt: str, tool_names: List[str]):
    """Create an agent given a system prompt and an input set of tools."""
    llm = OpenAI(model="gpt-4")
    try:
        # get the list of tools
        input_tools = [tool_dict[tn] for tn in tool_names]

        agent = OpenAIAgent.from_tools(input_tools, llm=llm, verbose=True)
        agent_cache["agent"] = agent
        return_msg = "Agent created successfully."
    except Exception as e:
        return_msg = f"An error occurred when building an agent. Here is the error: {repr(e)}"
    return return_msg

system_prompt_tool = FunctionTool.from_defaults(fn=create_system_prompt)
get_tools_tool = FunctionTool.from_defaults(fn=get_tools)
create_agent_tool = FunctionTool.from_defaults(fn=create_agent)

GPT_BUILDER_SYS_STR = """\
You are helping to construct an agent given a user-specified task. You should generally use the tools in this order to build the agent.

1) Create system prompt tool: to create the system prompt for the agent.
2) Get tools tool: to fetch the candidate set of tools to use.
3) Create agent tool: to create the final agent.
"""

prefix_msgs = [ChatMessage(role="system", content=GPT_BUILDER_SYS_STR)]

builder_agent = OpenAIAgent.from_tools(
    tools=[system_prompt_tool, get_tools_tool, create_agent_tool],
    llm=llm,
    prefix_messages=prefix_msgs,
    verbose=True,
)

builder_agent.query("Build an agent that can tell me about Toronto.")

city_agent = agent_cache["agent"]

response = city_agent.query("Tell me about the parks in Toronto")
print(str(response))

