#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/output_parsing/openai_pydantic_program.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # OpenAI Pydantic Program

# This guide shows you how to generate structured data with [new OpenAI API](https://openai.com/blog/function-calling-and-other-api-updates) via LlamaIndex. The user just needs to specify a Pydantic object.
# 
# We demonstrate two settings:
# - Extraction into an `Album` object (which can contain a list of Song objects)
# - Extraction into a `DirectoryTree` object (which can contain recursive Node objects)

# ## Extraction into `Album`
# 
# This is a simple example of parsing an output into an `Album` schema, which can contain multiple songs.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from pydantic import BaseModel
from typing import List

from llama_index.program import OpenAIPydanticProgram

# Define output schema

class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int

class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]

# Define openai pydantic program

prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""
program = OpenAIPydanticProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    verbose=True,
)

# Run program to get structured output.  

output = program(movie_name="The Shining")

# The output is a valid Pydantic object that we can then use to call functions/APIs. 

output

# ## Extracting List of `Album` (with Parallel Function Calling)

# With the latest [parallel function calling](https://platform.openai.com/docs/guides/function-calling/parallel-function-calling) feature from OpenAI, we can simultaneously extract multiple structured data from a single prompt!

# To do this, we need to:
# 1. pick one of the latest models (e.g. `gpt-3.5-turbo-1106`), and 
# 2. set `allow_multiple` to True in our `OpenAIPydanticProgram` (if not, it will only return the first object, and raise a warning).

from llama_index.llms import OpenAI

prompt_template_str = """\
Generate 4 albums about spring, summer, fall, and winter.
"""
program = OpenAIPydanticProgram.from_defaults(
    output_cls=Album,
    llm=OpenAI(model="gpt-3.5-turbo-1106"),
    prompt_template_str=prompt_template_str,
    allow_multiple=True,
    verbose=True,
)

output = program()

# The output is a list of valid Pydantic object.

output

# ## Extraction into `Album` (Streaming)
# 
# We also support streaming a list of objects through our `stream_list` function.
# 
# Full credits to this idea go to `openai_function_call` repo: https://github.com/jxnl/openai_function_call/tree/main/examples/streaming_multitask

prompt_template_str = "{input_str}"
program = OpenAIPydanticProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    verbose=False,
)

output = program.stream_list(input_str="make up 5 random albums")
for obj in output:
    print(obj.json(indent=2))

# ## Extraction into `DirectoryTree` object
# 
# This is directly inspired by jxnl's awesome repo here: https://github.com/jxnl/openai_function_call.
# 
# That repository shows how you can use OpenAI's function API to parse recursive Pydantic objects. The main requirement is that you want to "wrap" a recursive Pydantic object with a non-recursive one.
# 
# Here we show an example in a "directory" setting, where a `DirectoryTree` object wraps recursive `Node` objects, to parse a file structure.

# NOTE: defining recursive objects in a notebook causes errors
from directory import DirectoryTree, Node

DirectoryTree.schema()

program = OpenAIPydanticProgram.from_defaults(
    output_cls=DirectoryTree,
    prompt_template_str="{input_str}",
    verbose=True,
)

input_str = """
root
├── folder1
│   ├── file1.txt
│   └── file2.txt
└── folder2
    ├── file3.txt
    └── subfolder1
        └── file4.txt
"""

output = program(input_str=input_str)

# The output is a full DirectoryTree structure with recursive `Node` objects.

output

