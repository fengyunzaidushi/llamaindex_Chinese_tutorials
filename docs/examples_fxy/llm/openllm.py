#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/llm/openllm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # OpenLLM
# 
# There are two ways to interface with LLMs from [OpenLLM](https://github.com/bentoml/OpenLLM).
# 
# - Through [`openllm`](https://github.com/bentoml/OpenLLM) package if you want to run locally:
#   use `llama_index.llms.OpenLLM`
# - If there is a running OpenLLM Server, then it will wraps [openllm-client](https://github.com/bentoml/OpenLLM/tree/main/openllm-client):
#   use `llama_index.llms.OpenLLMAPI`
# 
# There are _many_ possible permutations of these two, so this notebook only details a few.
# See [OpenLLM's README](https://github.com/bentoml/OpenLLM) for more information

# 
# - `openllm[vllm]` is needed for `OpenLLM` if you have access to GPU, otherwise `openllm`
# - `openllm-client` is needed for `OpenLLMAPI`
# - The quotes are needed for Z shell (`zsh`)

#('pip install "openllm"  # use \'openllm[vllm]\' if you have access to GPU')

# Now that we're set up, let's play around:

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
from typing import List, Optional

from llama_index.llms import OpenLLM, OpenLLMAPI
from llama_index.llms import ChatMessage

os.environ[
    "OPENLLM_ENDPOINT"
] = "na"  # Change this to a remote server that you might run OpenLLM at.

# This uses https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha
# downloaded (if first invocation) to the local Hugging Face model cache,
# and actually runs the model on your local machine's hardware
local_llm = OpenLLM("HuggingFaceH4/zephyr-7b-alpha")

# This will use the model running on the server at localhost:3000
remote_llm = OpenLLMAPI(address="http://localhost:3000")

# Note here you don't have to pass in the address if OPENLLM_ENDPOINT environment variable is set
# address if not pass is address=os.getenv("OPENLLM_ENDPOINT")
remote_llm = OpenLLMAPI()

# Underlying a completion with `OpenLLM` supports continuous batching with [vLLM](https://vllm.ai/)

completion_response = remote_llm.complete("To infinity, and")
print(completion_response)

# `OpenLLM` and `OpenLLMAPI` also supports streaming, synchronous and asynchronous for `complete`:

for it in remote_llm.stream_complete(
    "The meaning of time is", max_new_tokens=128
):
    print(it, end="", flush=True)

# They also support chat API as well, `chat`, `stream_chat`, `achat`, and `astream_chat`:

async for it in remote_llm.astream_chat(
    [
        ChatMessage(
            role="system", content="You are acting as Ernest Hemmingway."
        ),
        ChatMessage(role="user", content="Hi there!"),
        ChatMessage(role="assistant", content="Yes?"),
        ChatMessage(role="user", content="What is the meaning of life?"),
    ]
):
    print(it.message.content, flush=True, end="")

