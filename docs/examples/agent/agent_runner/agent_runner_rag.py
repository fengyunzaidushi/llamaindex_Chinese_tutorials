#!/usr/bin/env python
# coding: utf-8

# # Controllable Agents for RAG
# 
# Adding agentic capabilities on top of your RAG pipeline can allow you to reason over much more complex questions.
# 
# But a big pain point for agents is the **lack of steerability/transparency**. An agent may tackle a user query through chain-of-thought/planning, which requires repeated calls to an LLM. During this process it can be hard to inspect what's going on, or stop/correct execution in the middle.
# 
# This notebook shows you how to use our brand-new lower-level agent API, which allows controllable step-wise execution, on top of a RAG pipeline.
# 
# We showcase this over Wikipedia documents.

#('pip install llama-index')

# ## Setup Data
# 
# Here we load a simple dataset of different cities from Wikipedia.

from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.schema import IndexNode
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.llms import OpenAI

wiki_titles = [
    "Toronto",
    "Seattle",
    "Chicago",
    "Boston",
    "Houston",
]

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

# Define LLM + Service Context + Callback Manager

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

# ## Setup Agent
# 

# ### Define Toolset
# 
# Each tool here corresponds to a simple top-k RAG pipeline over a single document / Wikipedia page.

from llama_index.agent import OpenAIAgent
from llama_index import load_index_from_storage, StorageContext
from llama_index.node_parser import SentenceSplitter
import os

node_parser = SentenceSplitter()

# Build agents dictionary
query_engine_tools = []

for idx, wiki_title in enumerate(wiki_titles):
    nodes = node_parser.get_nodes_from_documents(city_docs[wiki_title])

    if not os.path.exists(f"./data/{wiki_title}"):
        # build vector index
        vector_index = VectorStoreIndex(nodes, service_context=service_context)
        vector_index.storage_context.persist(
            persist_dir=f"./data/{wiki_title}"
        )
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./data/{wiki_title}"),
            service_context=service_context,
        )
    # define query engines
    vector_query_engine = vector_index.as_query_engine()

    # define tools
    query_engine_tools.append(
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{wiki_title}",
                description=(
                    "Useful for questions related to specific aspects of"
                    f" {wiki_title} (e.g. the history, arts and culture,"
                    " sports, demographics, or more)."
                ),
            ),
        )
    )

# ### Setup OpenAI Agent
# 
# We setup an OpenAI Agent through its components: an AgentRunner as well as an `OpenAIAgentWorker`.

from llama_index.agent import AgentRunner, OpenAIAgentWorker, OpenAIAgent
from llama_index.agent.openai.step import OpenAIAgentWorker

openai_step_engine = OpenAIAgentWorker.from_tools(
    query_engine_tools, llm=llm, verbose=True
)
agent = AgentRunner(openai_step_engine)
# # alternative
# agent = OpenAIAgent.from_tools(query_engine_tools, llm=llm, verbose=True)

# ## Run Some Queries
# 
# We now demonstrate the capabilities of our step-wise agent framework. 
# 
# We show how it can handle complex queries, both e2e as well as step by step. 
# 
# We can then show how we can steer the outputs.

# ### Out of the box

response = agent.chat(
    "Tell me about the demographics of Houston, and compare that with the demographics of Chicago"
)

print(str(response))

# list the task and steps for visibility
tasks = agent.list_tasks()
print(f"Task ID: {tasks[-1].task.task_id}")
completed_steps = agent.get_completed_steps(tasks[-1].task.task_id)
print(f"Number of steps: {len(completed_steps)}")

# ### Test Step-Wise Execution
# 
# We now break this query down into steps. We first create a task object from the user query.
# 
# We can then start running through steps - or even interjecting our own.

# start task
task = agent.create_task(
    "Tell me about the demographics of Houston, and compare that with the demographics of Chicago?"
)

# This returns a `Task` object, which contains the `input`, additional state in `extra_state`, and other fields.
# 
# Now let's try executing a single step of this task.

step_output = agent.run_step(task.task_id)

# When we inspect the logs and the output, we see that the first part was executed - the demographics of Houston.

completed_steps = agent.get_completed_steps(task.task_id)
print(f"Num completed for task {task.task_id}: {len(completed_steps)}")

# We can also take a look at the upcoming step.
# 
# **NOTE**: Currently the input is not shown, since execution of a step purely depends on internal memory. This is something we're working on!

upcoming_steps = agent.get_upcoming_steps(task.task_id)
print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")
upcoming_steps[0]

# If you wanted to pause execution now, you can - you can take the intermediate results without completing the agent flow!
# 
# **NOTE**: The `memory` of the agent (`agent.memory`) isn't modified until the task is complete and committed - so if you pause now, the memory won't be committed. This is good in case the execution fails.
# 
# Let's run the next two steps.

step_output = agent.run_step(task.task_id)

step_output = agent.run_step(task.task_id)
print(step_output.is_last)

# Since the steps look good, we are now ready to call `finalize_response`, get back our response.
# 
# This will also commit the task execution to the `memory` object present in our `agent_runner`. We can inspect it.

response = agent.finalize_response(task.task_id)

print(str(response))

# ##
# 
# We can inspect current and previous tasks and steps.
# 
# This gives you greater transparency into what the agent has processed!

tasks = agent.list_tasks()
print(len(tasks))

task_state = tasks[-1]
steps = agent.get_completed_steps(task_state.task.task_id)
print(len(steps))

