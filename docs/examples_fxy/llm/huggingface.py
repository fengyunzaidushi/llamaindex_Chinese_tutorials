#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/llm/huggingface.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Hugging Face LLMs
# 
# There are many ways to interface with LLMs from [Hugging Face](https://huggingface.co/).
# Hugging Face itself provides several Python packages to enable access,
# which LlamaIndex wraps into `LLM` entities:
# 
# - The [`transformers`](https://github.com/huggingface/transformers) package:
#   use `llama_index.llms.HuggingFaceLLM`
# - The [Hugging Face Inference API](https://huggingface.co/inference-api),
#   [wrapped by `huggingface_hub[inference]`](https://github.com/huggingface/huggingface_hub):
#   use `llama_index.llms.HuggingFaceInferenceAPI`
# 
# There are _many_ possible permutations of these two, so this notebook only details a few.
# Let's use Hugging Face's [Text Generation task](https://huggingface.co/tasks/text-generation) as our example.

# 
# - `transformers[torch]` is needed for `HuggingFaceLLM`
# - `huggingface_hub[inference]` is needed for `HuggingFaceInferenceAPI`
# - The quotes are needed for Z shell (`zsh`)

#('pip install "transformers[torch]" "huggingface_hub[inference]"')

# Now that we're set up, let's play around:

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
from typing import List, Optional

from llama_index.llms import HuggingFaceInferenceAPI, HuggingFaceLLM

# SEE: https://huggingface.co/docs/hub/security-tokens
# We just need a token with read permissions for this demo
HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")
# NOTE: None default will fall back on Hugging Face's token storage
# when this token gets used within HuggingFaceInferenceAPI

# This uses https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha
# downloaded (if first invocation) to the local Hugging Face model cache,
# and actually runs the model on your local machine's hardware
locally_run = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-alpha")

# This will use the same model, but run remotely on Hugging Face's servers,
# accessed via the Hugging Face Inference API
# Note that using your token will not charge you money,
# the Inference API is free it just has rate limits
remotely_run = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-alpha", token=HF_TOKEN
)

# Or you can skip providing a token, using Hugging Face Inference API anonymously
remotely_run_anon = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-alpha"
)

# If you don't provide a model_name to the HuggingFaceInferenceAPI,
# Hugging Face's recommended model gets used (thanks to huggingface_hub)
remotely_run_recommended = HuggingFaceInferenceAPI(token=HF_TOKEN)

# Underlying a completion with `HuggingFaceInferenceAPI` is Hugging Face's
# [Text Generation task](https://huggingface.co/tasks/text-generation).

completion_response = remotely_run_recommended.complete("To infinity, and")
print(completion_response)

# If you are modifying the LLM, you should also change the global tokenizer to match!

from llama_index import set_global_tokenizer
from transformers import AutoTokenizer

set_global_tokenizer(
    AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha").encode
)

# If you're curious, other Hugging Face Inference API tasks wrapped are:
# 
# - `llama_index.llms.HuggingFaceInferenceAPI.chat`: [Conversational task](https://huggingface.co/tasks/conversational)
# - `llama_index.embeddings.HuggingFaceInferenceAPIEmbedding`: [Feature Extraction task](https://huggingface.co/tasks/feature-extraction)
# 
# And yes, Hugging Face embedding models are supported with:
# 
# - `transformers[torch]`: wrapped by `HuggingFaceEmbedding`
# - `huggingface_hub[inference]`: wrapped by `HuggingFaceInferenceAPIEmbedding`
# 
# Both of the above two subclass `llama_index.embeddings.base.BaseEmbedding`.
