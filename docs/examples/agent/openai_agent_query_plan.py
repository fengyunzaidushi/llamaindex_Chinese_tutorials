#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/agent/openai_agent_query_plan.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # OpenAI Agent Query Planning

# to do advanced query planning, all through a single tool! 
# 
# The `QueryPlanTool` is designed to work well with the OpenAI Function API. The tool takes in a set of other tools as input.
# The tool function signature contains of a QueryPlan Pydantic object, which can in turn contain a DAG of QueryNode objects defining a compute graph.
# The agent is responsible for defining this graph through the function signature when calling the tool. The tool itself executes the DAG over any corresponding tools.
# 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# # uncomment to turn on logging
# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    GPTVectorStoreIndex,
)
from llama_index.response.pprint_utils import pprint_response
from llama_index.llms import OpenAI

llm = OpenAI(temperature=0, model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)

# ## Download Data

#("mkdir -p 'data/10q/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10q/uber_10q_march_2022.pdf' -O 'data/10q/uber_10q_march_2022.pdf'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10q/uber_10q_june_2022.pdf' -O 'data/10q/uber_10q_june_2022.pdf'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10q/uber_10q_sept_2022.pdf' -O 'data/10q/uber_10q_sept_2022.pdf'")

# ## Load data

march_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_march_2022.pdf"]
).load_data()
june_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_june_2022.pdf"]
).load_data()
sept_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_sept_2022.pdf"]
).load_data()

# ## Build indices
# 
# We build a vector index / query engine over each of the documents (March, June, September).

march_index = GPTVectorStoreIndex.from_documents(march_2022)
june_index = GPTVectorStoreIndex.from_documents(june_2022)
sept_index = GPTVectorStoreIndex.from_documents(sept_2022)

march_engine = march_index.as_query_engine(
    similarity_top_k=3, service_context=service_context
)
june_engine = june_index.as_query_engine(
    similarity_top_k=3, service_context=service_context
)
sept_engine = sept_index.as_query_engine(
    similarity_top_k=3, service_context=service_context
)

# ## OpenAI Function Agent with a Query Plan Tool
# 
# Use OpenAIAgent, built on top of the OpenAI tool use interface.
# 
# Feed it our QueryPlanTool, which is a Tool that takes in other tools. And the agent to generate a query plan DAG over these tools.

from llama_index.tools import QueryEngineTool

query_tool_sept = QueryEngineTool.from_defaults(
    query_engine=sept_engine,
    name="sept_2022",
    description=(
        f"Provides information about Uber quarterly financials ending"
        f" September 2022"
    ),
)
query_tool_june = QueryEngineTool.from_defaults(
    query_engine=june_engine,
    name="june_2022",
    description=(
        f"Provides information about Uber quarterly financials ending June"
        f" 2022"
    ),
)
query_tool_march = QueryEngineTool.from_defaults(
    query_engine=march_engine,
    name="march_2022",
    description=(
        f"Provides information about Uber quarterly financials ending March"
        f" 2022"
    ),
)

# define query plan tool
from llama_index.tools import QueryPlanTool
from llama_index import get_response_synthesizer

response_synthesizer = get_response_synthesizer(
    service_context=service_context
)
query_plan_tool = QueryPlanTool.from_defaults(
    query_engine_tools=[query_tool_sept, query_tool_june, query_tool_march],
    response_synthesizer=response_synthesizer,
)

query_plan_tool.metadata.to_openai_tool()  # to_openai_function() deprecated

from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI

agent = OpenAIAgent.from_tools(
    [query_plan_tool],
    max_function_calls=10,
    llm=OpenAI(temperature=0, model="gpt-4-0613"),
    verbose=True,
)

response = agent.query("What were the risk factors in sept 2022?")

from llama_index.tools.query_plan import QueryPlan, QueryNode

query_plan = QueryPlan(
    nodes=[
        QueryNode(
            id=1,
            query_str="risk factors",
            tool_name="sept_2022",
            dependencies=[],
        )
    ]
)

QueryPlan.schema()

response = agent.query(
    "Analyze Uber revenue growth in March, June, and September"
)

print(str(response))

response = agent.query(
    "Analyze changes in risk factors in march, june, and september for Uber"
)

print(str(response))

# response = agent.query("Analyze both Uber revenue growth and risk factors over march, june, and september")

print(str(response))

response = agent.query(
    "First look at Uber's revenue growth and risk factors in March, "
    + "then revenue growth and risk factors in September, and then compare and"
    " contrast the two documents?"
)

response

