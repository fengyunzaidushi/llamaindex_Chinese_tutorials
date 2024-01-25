#!/usr/bin/env python
# coding: utf-8

# # Perplexity
# 
# Before we get started, make sure you install llama_index

#('pip install llama-index')

# ## Setup LLM
# 
# As of Nov 14, 2023 - the following models are supported with the Perplexity LLM class in LLaMa Index:
# 
# | Model | Context Length | Model Type |
# |-------|----------------|------------|
# | codellama-34b-instruct | 16384 | Chat Completion |
# | llama-2-13b-chat | 4096 | Chat Completion |
# | llama-2-70b-chat | 4096 | Chat Completion |
# | mistral-7b-instruct | 4096 [1] | Chat Completion |
# | openhermes-2-mistral-7b | 4096 [1] | Chat Completion |
# | openhermes-2.5-mistral-7b | 4096 [1] | Chat Completion |
# | replit-code-v1.5-3b | 4096 | Text Completion |
# | pplx-7b-chat-alpha | 4096 | Chat Completion |
# | pplx-70b-chat-alpha | 4096 | Chat Completion |
# 
# [1] Context length of mistral-7b-instruct and openhermes-2-mistral-7b will be increased to 32k tokens (see perplexity roadmap).
# 
# You can find the latest supported models here - https://docs.perplexity.ai/docs/model-cards \
# Rate limits are found here - https://docs.perplexity.ai/docs/rate-limits

from llama_index.llms import Perplexity

pplx_api_key = "your-perplexity-api-key"

llm = Perplexity(
    api_key=pplx_api_key, model="mistral-7b-instruct", temperature=0.5
)

from llama_index.llms import ChatMessage

messages_dict = [
    {"role": "system", "content": "Be precise and concise."},
    {"role": "user", "content": "Tell me 5 sentences about Perplexity."},
]
messages = [ChatMessage(**msg) for msg in messages_dict]

# ## Chat

response = llm.chat(messages)
print(response)

# ## Async Chat

response = await llm.achat(messages)
print(response)

# ## Stream Chat

resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")

# ## Async Stream Chat

resp = await llm.astream_chat(messages)
async for delta in resp:
    print(delta.delta, end="")

