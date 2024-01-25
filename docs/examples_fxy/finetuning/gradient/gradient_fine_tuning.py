#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/finetuning/gradient/gradient_fine_tuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Fine Tuning Nous-Hermes-2 With Gradient and LlamaIndex
# 

#('pip install llama-index gradientai -q')

import os
from llama_index.llms import GradientBaseModelLLM
from llama_index.finetuning.gradient.base import GradientFinetuneEngine

os.environ["GRADIENT_ACCESS_TOKEN"] = ""
os.environ["GRADIENT_WORKSPACE_ID"] = ""

questions = [
    "Where do foo-bears live?",
    "What do foo-bears look like?",
    "What do foo-bears eat?",
]

prompts = list(
    f"<s> ##
)

base_model_slug = "nous-hermes2"
base_model_llm = GradientBaseModelLLM(
    base_model_slug=base_model_slug, max_tokens=100
)

base_model_responses = list(base_model_llm.complete(p).text for p in prompts)

finetune_engine = GradientFinetuneEngine(
    base_model_slug=base_model_slug,
    name="my test finetune engine model adapter",
    data_path="data.jsonl",
)

# warming up with the first epoch can lead to better results, our current optimizers are momentum based
epochs = 2
for i in range(epochs):
    finetune_engine.finetune()
fine_tuned_model = finetune_engine.get_finetuned_model(max_tokens=100)

fine_tuned_model_responses = list(
    fine_tuned_model.complete(p).text for p in prompts
)
fine_tuned_model._model.delete()

for i, q in enumerate(questions):
    print(f"Question: {q}")
    print(f"Base: {base_model_responses[i]}")
    print(f"Fine tuned: {fine_tuned_model_responses[i]}")
    print()

