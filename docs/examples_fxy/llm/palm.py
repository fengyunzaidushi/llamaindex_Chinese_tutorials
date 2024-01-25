#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/palm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # PaLM 
# 

# 
# We use the `text-bison-001` model by default.

# ### Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

#('pip install -q google-generativeai')

import pprint
import google.generativeai as palm

palm_api_key = ""

palm.configure(api_key=palm_api_key)

# ### Define Model

models = [
    m
    for m in palm.list_models()
    if "generateText" in m.supported_generation_methods
]
model = models[0].name
print(model)

# ### Start using our `PaLM` LLM abstraction!

from llama_index.llms.palm import PaLM

model = PaLM(api_key=palm_api_key)

model.complete(prompt)

