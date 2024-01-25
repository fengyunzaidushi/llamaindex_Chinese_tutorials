#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/llama_2_rap_battle.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # 🦙 x 🦙 Rap Battle

# What happens when 2 🦙 (13B vs. 70B) have a rap battle?

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ### Setup

from llama_index.llms import Replicate
from llama_index.llms.llama_utils import messages_to_prompt

llm_13b = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    context_window=4096,
    messages_to_prompt=messages_to_prompt,  # override message representation for llama 2
)
llm_70b = Replicate(
    model="replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48",
    context_window=4096,
    messages_to_prompt=messages_to_prompt,  # override message representation for llama 2
)

from llama_index.chat_engine import SimpleChatEngine
from llama_index.memory import ChatMemoryBuffer
from llama_index.llms import ChatMessage

bot_70b = SimpleChatEngine(
    llm=llm_70b,
    memory=ChatMemoryBuffer.from_defaults(llm=llm_70b),
    prefix_messages=[
        ChatMessage(
            role="system", content="You are a rapper with an ENTJ personality"
        )
    ],
)
bot_13b = SimpleChatEngine(
    llm=llm_13b,
    memory=ChatMemoryBuffer.from_defaults(llm=llm_13b),
    prefix_messages=[
        ChatMessage(
            role="system", content="You are a rapper with an INFP personality"
        )
    ],
)

# ### Let the rap battle begin!

n_turns = 2

message = "Please introduce yourself and pick a topic to start a rap battle"
for _ in range(n_turns):
    message = bot_70b.chat(message).response
    print("==============================")
    print("🦙 70B: ", message)

    message = bot_13b.chat(message).response
    print("==============================")
    print("🦙 13B: ", message)

