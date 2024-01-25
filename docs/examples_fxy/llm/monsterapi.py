#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/monsterapi.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Monster API LLM Integration into LLamaIndex
# 
# MonsterAPI Hosts wide range of popular LLMs as inference service and this notebook serves as a tutorial about how to use llama-index to access MonsterAPI LLMs.
# 
# 
# Check us out here: https://monsterapi.ai/
# 

#('python3 -m pip install llama-index --quiet -y')
#('python3 -m pip install monsterapi --quiet')
#('python3 -m pip install sentence_transformers --quiet')

# Import required modules

import os

from llama_index.llms import MonsterLLM
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

# ### Set Monster API Key env variable
# 
# Sign up on [MonsterAPI](https://monsterapi.ai/signup?utm_source=llama-index-colab&utm_medium=referral) and get a free auth key. Paste it below:

os.environ["MONSTER_API_KEY"] = ""

# ## Basic Usage Pattern

# Set the model

model = "llama2-7b-chat"

llm = MonsterLLM(model=model, temperature=0.75)

# ### Completion Example

result = llm.complete("Who are you?")
print(result)

# ### Chat Example

from llama_index.llms import ChatMessage

# Construct mock Chat history
history_message = ChatMessage(
    **{
        "role": "user",
        "content": (
            "When asked 'who are you?' respond as 'I am qblocks llm model'"
            " everytime."
        ),
    }
)
current_message = ChatMessage(**{"role": "user", "content": "Who are you?"})

response = llm.chat([history_message, current_message])
print(response)

# ##RAG Approach to import external knowledge into LLM as context
# 
# Source Paper: https://arxiv.org/pdf/2005.11401.pdf
# 
# Retrieval-Augmented Generation (RAG) is a method that uses a combination of pre-defined rules or parameters (non-parametric memory) and external information from the internet (parametric memory) to generate responses to questions or create new ones. By lever

#('python3 -m pip install pypdf --quiet')

# Lets try to augment our LLM with RAG source paper PDF as external information.
# Lets download the pdf into data dir

#('rm -r ./data')
#('mkdir -p data&&cd data&&curl \'https://arxiv.org/pdf/2005.11401.pdf\' -o "RAG.pdf"')

# Load the document

documents = SimpleDirectoryReader("./data").load_data()

llm = MonsterLLM(model=model, temperature=0.75, context_window=1024)
service_context = ServiceContext.from_defaults(
    chunk_size=1024, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
)

# Create embedding store and create index

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
query_engine = index.as_query_engine()

# Actual LLM output without RAG:

llm.complete("What is Retrieval-Augmented Generation?")

# LLM Output with RAG

response = query_engine.query("What is Retrieval-Augmented Generation?")
print(response)

