#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/multi_modal/openai_multi_modal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# # Multi-Modal LLM using OpenAI GPT-4V model for image reasoning
# 

# 
# We also show several functions we are now supporting for OpenAI GPT4V LLM:
# * `complete` (both sync and async): for a single prompt and list of images
# * `chat` (both sync and async): for multiple chat messages
# * `stream complete` (both sync and async): for steaming output of complete
# * `stream chat` (both sync and async): for steaming output of chat

#('pip install openai matplotlib')

# ##  Use GPT4V to understand Images from URLs

import os

OPENAI_API_TOKEN = "sk-"  # Your OpenAI API token here
os.environ["OPENAI_API_TOKEN"] = OPENAI_API_TOKEN

# #

# ## 

from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from llama_index.multi_modal_llms.generic_utils import (
    load_image_urls,
)

image_urls = [
    # "https://www.visualcapitalist.com/wp-content/uploads/2023/10/US_Mortgage_Rate_Surge-Sept-11-1.jpg",
    # "https://www.sportsnet.ca/wp-content/uploads/2023/11/CP1688996471-1040x572.jpg",
    "https://res.cloudinary.com/hello-tickets/image/upload/c_limit,f_auto,q_auto,w_1920/v1640835927/o3pfl41q7m5bj8jardk0.jpg",
    # "https://www.cleverfiles.com/howto/wp-content/uploads/2018/03/minion.jpg",
]

image_documents = load_image_urls(image_urls)

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=300
)

from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

img_response = requests.get(image_urls[0])
print(image_urls[0])
img = Image.open(BytesIO(img_response.content))
plt.imshow(img)

# ### Complete a prompt with a bunch of images

complete_response = openai_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)

print(complete_response)

# ### Steam Complete a prompt with a bunch of images

stream_complete_response = openai_mm_llm.stream_complete(
    prompt="give me more context for this image",
    image_documents=image_documents,
)

for r in stream_complete_response:
    print(r.delta, end="")

# ### Chat through a list of chat messages

from llama_index.multi_modal_llms.openai_utils import (
    generate_openai_multi_modal_chat_message,
)

chat_msg_1 = generate_openai_multi_modal_chat_message(
    prompt="Describe the images as an alternative text",
    role="user",
    image_documents=image_documents,
)

chat_msg_2 = generate_openai_multi_modal_chat_message(
    prompt="The image is a graph showing the surge in US mortgage rates. It is a visual representation of data, with a title at the top and labels for the x and y-axes. Unfortunately, without seeing the image, I cannot provide specific details about the data or the exact design of the graph.",
    role="assistant",
)

chat_msg_3 = generate_openai_multi_modal_chat_message(
    prompt="can I know more?",
    role="user",
)

chat_messages = [chat_msg_1, chat_msg_2, chat_msg_3]
chat_response = openai_mm_llm.chat(
    # prompt="Describe the images as an alternative text",
    messages=chat_messages,
)

for msg in chat_messages:
    print(msg.role, msg.content)

print(chat_response)

# ### Stream Chat through a list of chat messages

stream_chat_response = openai_mm_llm.stream_chat(
    # prompt="Describe the images as an alternative text",
    messages=chat_messages,
)

for r in stream_chat_response:
    print(r.delta, end="")

# ### Async Complete

response_acomplete = await openai_mm_llm.acomplete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)

print(response_acomplete)

# ### Async Steam Complete

response_astream_complete = await openai_mm_llm.astream_complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)

async for delta in response_astream_complete:
    print(delta.delta, end="")

# ### Async Chat

achat_response = await openai_mm_llm.achat(
    messages=chat_messages,
)

print(achat_response)

# ### Async stream Chat

astream_chat_response = await openai_mm_llm.astream_chat(
    messages=chat_messages,
)

async for delta in astream_chat_response:
    print(delta.delta, end="")

# ## Complete with Two images

image_urls = [
    "https://www.visualcapitalist.com/wp-content/uploads/2023/10/US_Mortgage_Rate_Surge-Sept-11-1.jpg",
    "https://www.sportsnet.ca/wp-content/uploads/2023/11/CP1688996471-1040x572.jpg",
    # "https://res.cloudinary.com/hello-tickets/image/upload/c_limit,f_auto,q_auto,w_1920/v1640835927/o3pfl41q7m5bj8jardk0.jpg",
    # "https://www.cleverfiles.com/howto/wp-content/uploads/2018/03/minion.jpg",
]

image_documents_1 = load_image_urls(image_urls)

response_multi = openai_mm_llm.complete(
    prompt="is there any relationship between those images?",
    image_documents=image_documents_1,
)
print(response_multi)

# ##  Use GPT4V to understand images from local files

from llama_index import SimpleDirectoryReader

# put your local directore here
image_documents = SimpleDirectoryReader("./images_wiki").load_data()

response = openai_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)

from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("./images_wiki/3.jpg")
plt.imshow(img)

print(response)

