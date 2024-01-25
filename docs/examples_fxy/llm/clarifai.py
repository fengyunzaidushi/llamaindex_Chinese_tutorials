#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/llm/clarifai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Clarifai LLM
# 
# ## Example notebook to call different LLM models using clarifai

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

#('pip install clarifai')

# Set clarifai PAT as environment variable.

import os

os.environ["CLARIFAI_PAT"] = "<YOUR CLARIFAI PAT>"

# Import clarifai package

from llama_index.llms.clarifai import Clarifai

# Explore various models according to your prefrence from
# [Our Models page](https://clarifai.com/explore/models?filterData=%5B%7B%22field%22%3A%22use_cases%22%2C%22value%22%3A%5B%22llm%22%5D%7D%5D&page=2&perPage=24)

# Example parameters
params = dict(
    user_id="clarifai",
    app_id="ml",
    model_name="llama2-7b-alternative-4k",
    model_url=(
        "https://clarifai.com/clarifai/ml/models/llama2-7b-alternative-4k"
    ),
)

# Method:1 using model_url parameter
llm_model = Clarifai(model_url=params["model_url"])

# Method:2 using model_name, app_id & user_id parameters
llm_model = Clarifai(
    model_name=params["model_name"],
    app_id=params["app_id"],
    user_id=params["user_id"],
)

# Call `complete` function

llm_reponse = llm_model.complete(
    prompt="write a 10 line rhyming poem about science"
)

print(llm_reponse)

# Call `chat` function

from llama_index.llms import ChatMessage

messages = [
    ChatMessage(role="user", content="write about climate change in 50 lines")
]
Response = llm_model.chat(messages)

print(Response)

# ### Using Inference parameters
# Alternatively you can call models with inference parameters.

# Here is an inference parameter example for GPT model.
inference_params = dict(temperature=str(0.3), max_tokens=20)

llm_reponse = llm_model.complete(
    prompt="What is nuclear fission and fusion?",
    inference_params=params,
)

messages = [ChatMessage(role="user", content="Explain about the big bang")]
Response = llm_model.chat(messages, inference_params=params)

