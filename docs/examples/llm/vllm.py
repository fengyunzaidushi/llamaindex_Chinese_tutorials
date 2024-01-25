#!/usr/bin/env python
# coding: utf-8

# 
# `pip install vllm` <br>
# or if you want to compile you can compile from <br>
# https://docs.vllm.ai/en/latest/getting_started/installation.html

# # Orca-7b Completion Example
# 

import os

os.environ["HF_HOME"] = "model/"

from llama_index.llms.vllm import Vllm

llm = Vllm(
    model="microsoft/Orca-2-7b",
    tensor_parallel_size=4,
    max_new_tokens=100,
    vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
)

llm.complete(
    ["[INST]You are a helpful assistant[/INST] What is a black hole ?"]
)

# # LLama-2-7b Completion Example
# 

llm = Vllm(
    model="codellama/CodeLlama-7b-hf",
    dtype="float16",
    tensor_parallel_size=4,
    temperature=0,
    max_new_tokens=100,
    vllm_kwargs={
        "swap_space": 1,
        "gpu_memory_utilization": 0.5,
        "max_model_len": 4096,
    },
)

llm.complete(["import socket\n\ndef ping_exponential_backoff(host: str):"])

# # mistral chat 7b Completion Example
# 

llm = Vllm(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    dtype="float16",
    tensor_parallel_size=4,
    temperature=0,
    max_new_tokens=100,
    vllm_kwargs={
        "swap_space": 1,
        "gpu_memory_utilization": 0.5,
        "max_model_len": 4096,
    },
)

llm.complete([" What is a black hole ?"])

# ## Completion Example

from llama_index.llms.vllm import VllmServer

llm = VllmServer(
    api_url="http://localhost:8000/generate", max_new_tokens=100, temperature=0
)

llm.complete("what is a black hole ?")

# ## Streaming Response

list(llm.stream_complete("what is a black hole"))[-1]

# # Api Response
# To setup the api you can follow the guide present here -> https://docs.vllm.ai/en/latest/serving/distributed_serving.html

# ## completion Response 

from llama_index.llms.vllm import VllmServer
from llama_index.llms import ChatMessage

llm = VllmServer(
    api_url="http://localhost:8000/generate", max_new_tokens=100, temperature=0
)

llm.complete("what is a black hole ?")

message = [ChatMessage(content="hello", author="user")]
llm.chat(message)

# ## Streaming Response

list(llm.stream_complete("what is a black hole"))[-1]

message = [ChatMessage(content="what is a black hole", author="user")]
[x for x in llm.stream_chat(message)][-1]

# ## Async Response

await llm.acomplete("What is a black hole")

await llm.achat(message)

[x async for x in await llm.astream_complete("what is a black hole")][-1]

[x for x in await llm.astream_chat(message)][-1]

