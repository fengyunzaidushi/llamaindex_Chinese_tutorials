#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/query_engine/json_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # JSON Query Engine
# The JSON query engine is useful for querying JSON documents that conform to a JSON schema.
# 
# This JSON schema is then used in the context of a prompt to convert a natural language query into a structured JSON Path query. This JSON Path query is then used to retrieve data to answer the given question.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# First, install the jsonpath-ng package which is used by default to parse & execute the JSONPath queries.
#('pip install jsonpath-ng')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import os
import openai

os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE"
openai.api_key = os.environ["OPENAI_API_KEY"]

from IPython.#display import Markdown, #display

# ### Let's start on a Toy JSON
# 
# Very simple JSON object containing data from a blog post site with user comments.
# 
# We will also provide a JSON schema (which we were able to generate by giving ChatGPT a sample of the JSON).
# 
# #### Advice
# Do make sure that you've provided a helpful `"description"` value for each of the fields in your JSON schema.
# 
# As you can see in the given example, the description for the `"username"` field mentions that usernames are lowercased. You'll see that this ends up being helpful for the LLM in producing the correct JSON path query.

# Test on some sample data
json_value = {
    "blogPosts": [
        {
            "id": 1,
            "title": "First blog post",
            "content": "This is my first blog post",
        },
        {
            "id": 2,
            "title": "Second blog post",
            "content": "This is my second blog post",
        },
    ],
    "comments": [
        {
            "id": 1,
            "content": "Nice post!",
            "username": "jerry",
            "blogPostId": 1,
        },
        {
            "id": 2,
            "content": "Interesting thoughts",
            "username": "simon",
            "blogPostId": 2,
        },
        {
            "id": 3,
            "content": "Loved reading this!",
            "username": "simon",
            "blogPostId": 2,
        },
    ],
}

# JSON Schema object that the above JSON value conforms to
json_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "description": "Schema for a very simple blog post app",
    "type": "object",
    "properties": {
        "blogPosts": {
            "description": "List of blog posts",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "description": "Unique identifier for the blog post",
                        "type": "integer",
                    },
                    "title": {
                        "description": "Title of the blog post",
                        "type": "string",
                    },
                    "content": {
                        "description": "Content of the blog post",
                        "type": "string",
                    },
                },
                "required": ["id", "title", "content"],
            },
        },
        "comments": {
            "description": "List of comments on blog posts",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "description": "Unique identifier for the comment",
                        "type": "integer",
                    },
                    "content": {
                        "description": "Content of the comment",
                        "type": "string",
                    },
                    "username": {
                        "description": (
                            "Username of the commenter (lowercased)"
                        ),
                        "type": "string",
                    },
                    "blogPostId": {
                        "description": (
                            "Identifier for the blog post to which the comment"
                            " belongs"
                        ),
                        "type": "integer",
                    },
                },
                "required": ["id", "content", "username", "blogPostId"],
            },
        },
    },
    "required": ["blogPosts", "comments"],
}

from llama_index.indices.service_context import ServiceContext
from llama_index.llms import OpenAI
from llama_index.indices.struct_store import JSONQueryEngine

llm = OpenAI(model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)
nl_query_engine = JSONQueryEngine(
    json_value=json_value,
    json_schema=json_schema,
    service_context=service_context,
)
raw_query_engine = JSONQueryEngine(
    json_value=json_value,
    json_schema=json_schema,
    service_context=service_context,
    synthesize_response=False,
)

nl_response = nl_query_engine.query(
    "What comments has Jerry been writing?",
)
raw_response = raw_query_engine.query(
    "What comments has Jerry been writing?",
)

#display(
    Markdown(f"<h1>Natural language Response</h1><br><b>{nl_response}</b>")
)
#display(Markdown(f"<h1>Raw JSON Response</h1><br><b>{raw_response}</b>"))

# get the json path query string. Same would apply to raw_response
print(nl_response.metadata["json_path_response_str"])

