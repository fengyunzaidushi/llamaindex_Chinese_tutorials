#!/usr/bin/env python
# coding: utf-8

# # Github Issue Analysis

# ## Setup

# To use the github repo issue loader, you need to set your github token in the environment.  
# 
# See [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) for how to get a github token.  
# See [llama-hub](https://llama-hub-ui.vercel.app/l/github_repo_issues) for more details about the loader.

import os

os.environ["GITHUB_TOKEN"] = "<your github token>"

# ## Load Github Issue tickets

import os

from llama_hub.github_repo_issues import (
    GitHubRepositoryIssuesReader,
    GitHubIssuesClient,
)

github_client = GitHubIssuesClient()
loader = GitHubRepositoryIssuesReader(
    github_client,
    owner="jerryjliu",
    repo="llama_index",
    verbose=True,
)

docs = loader.load_data()

# Quick inspection

docs[10].text

docs[10].metadata

# ## Extract themes

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from pydantic import BaseModel
from typing import List
from tqdm.asyncio import asyncio

from llama_index.program import OpenAIPydanticProgram
from llama_index.llms import OpenAI
from llama_index.async_utils import batch_gather

prompt_template_str = """\
Here is a Github Issue ticket.

{ticket}

Please extract central themes and output a list of tags.\
"""

class TagList(BaseModel):
    """A list of tags corresponding to central themes of an issue."""

    tags: List[str]

program = OpenAIPydanticProgram.from_defaults(
    prompt_template_str=prompt_template_str,
    output_cls=TagList,
)

tasks = [program.acall(ticket=doc) for doc in docs]

output = await batch_gather(tasks, batch_size=10, verbose=True)

# ## [Optional] Save/Load Extracted Themes 

import pickle

with open("github_issue_analysis_data.pkl", "wb") as f:
    pickle.dump(tag_lists, f)

with open("github_issue_analysis_data.pkl", "rb") as f:
    tag_lists = pickle.load(f)
    print(f"Loaded tag lists for {len(tag_lists)} tickets")

# ## Summarize Themes

# Build prompt 

prompt = """
Here is a list of central themes (in the form of tags) extracted from a list of Github Issue tickets.
Tags for each ticket is separated by 2 newlines.

{tag_lists_str}

Please summarize the key takeaways and what we should prioritize to fix.
"""

tag_lists_str = "\n\n".join([str(tag_list) for tag_list in tag_lists])

prompt = prompt.format(tag_lists_str=tag_lists_str)

# Summarize with GPT-4

from llama_index.llms import OpenAI

response = OpenAI(model="gpt-4").stream_complete(prompt)

for r in response:
    print(r.delta, end="")

