#!/usr/bin/env python
# coding: utf-8

# # Step-wise, Controllable Agents
# 
# This notebook shows you how to use our brand-new lower-level agent API, which supports a host of functionalities beyond simply executing a user query to help you create tasks, iterate through steps, and control the inputs for each step.
# 
# ### High-Level Agent Architecture
# 
# Our "agents" are composed of `AgentRunner` objects that interact with `AgentWorkers`. `AgentRunner`s are orchestrators that store state (including conversational memory), create and maintain tasks, run steps through each task, and offer the user-facing, high-level interface for users to interact with.
# 
# `AgentWorker`s **control the step-wise execution of a Task**. Given an input step, an agent worker is responsible for generating the next step. They can be initialized with parameters and act upon state passed down from the Task/TaskStep objects, but do not inherently store state themselves. The outer `AgentRunner` is responsible for calling an `AgentWorker` and collecting/aggregating the results.
# 
# If you are building your own agent, you will likely want to create your own `AgentWorker`. See below for an example!
# 
# ### Notebook Walkthrough
# 
# This notebook shows you how to run step-wise execution and full-execution with agents. 
# - We show you how to do execution with OpenAIAgent (function calling)
# - We show you how to do execution with ReActAgent

#('pip install llama-index')

import json
from typing import Sequence, List

from llama_index.llms import OpenAI, ChatMessage
from llama_index.tools import BaseTool, FunctionTool

import nest_asyncio

nest_asyncio.apply()

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

tools = [multiply_tool, add_tool]

llm = OpenAI(model="gpt-3.5-turbo")

# ## Test OpenAI Agent
# 
# There's two main ways to initialize the agent.
# - **Option 1**: Initialize `OpenAIAgent`. This is a simple subclass of `AgentRunner` that bundles the `OpenAIAgentWorker` under the hood.
# - **Option 2**: Initialize `AgentRunner` with `OpenAIAgentWorker`. Here you import the modules and compose your own agent.
# 
# **NOTE**: The old OpenAIAgent can still be imported via `from llama_index.agent import OldOpenAIAgent`.

from llama_index.agent import AgentRunner, OpenAIAgentWorker, OpenAIAgent

# Option 1: Initialize OpenAIAgent
agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)

# # Option 2: Initialize AgentRunner with OpenAIAgentWorker
# openai_step_engine = OpenAIAgentWorker.from_tools(tools, llm=llm, verbose=True)
# agent = AgentRunner(openai_step_engine)

# ### Test E2E Chat
# 
# Here we re-demonstrate the end-to-end execution of a user task through the `chat()` function.
# 
# This will iterate step-wise until the agent is done with the current task.

agent.chat("Hi")

response = agent.chat("What is (121 * 3) + 42?")

response

# ### Test Step-Wise Execution
# 
# Now let's show the lower-level API in action. We do the same thing, but break this down into steps.

# start task
task = agent.create_task("What is (121 * 3) + 42?")

step_output = agent.run_step(task.task_id)

step_output

step_output = agent.run_step(task.task_id)

step_output = agent.run_step(task.task_id)

# #display final response
print(step_output.is_last)

# now that the step execution is done, we can finalize response
response = agent.finalize_response(task.task_id)
print(str(response))

# ## Test ReAct Agent
# 
# We do the same experiments, but with ReAct.

llm = OpenAI(model="gpt-4-1106-preview")

from llama_index.agent import AgentRunner, ReActAgentWorker, ReActAgent

# Option 1: Initialize OpenAIAgent
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

# # Option 2: Initialize AgentRunner with ReActAgentWorker
# react_step_engine = ReActAgentWorker.from_tools(tools, llm=llm, verbose=True)
# agent = AgentRunner(react_step_engine)

agent.chat("Hi")

response = agent.chat("What is (121 * 3) + 42?")

response

# start task
task = agent.create_task("What is (121 * 3) + 42?")

step_output = agent.run_step(task.task_id)

step_output.output

step_output = agent.run_step(task.task_id)

step_output.output

step_output = agent.run_step(task.task_id)

step_output.output

# ### List Out Tasks
# 
# There are 3 tasks, corresponding to the three runs above.

tasks = agent.list_tasks()
print(len(tasks))

task_state = tasks[-1]
task_state.task.input

# get completed steps
completed_steps = agent.get_completed_steps(task_state.task.task_id)

len(completed_steps)

completed_steps[0]

for idx in range(len(completed_steps)):
    print(f"Step {idx}")
    print(f"Response: {completed_steps[idx].output.response}")
    print(f"Sources: {completed_steps[idx].output.sources}")

