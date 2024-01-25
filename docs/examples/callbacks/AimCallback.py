#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/callbacks/AimCallback.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Aim Callback
# 
# Aim is an easy-to-use & supercharged open-source AI metadata tracker it logs all your AI metadata (experiments, prompts, etc) enables a UI to compare & observe them and SDK to query them programmatically. For more please see the [Github page](https://github.com/aimhubio/aim).
# 

# 
# 
# **NOTE**: This is a beta feature. The usage within different classes and the API interface for the CallbackManager and AimCallback may change!

# ## Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.callbacks import CallbackManager, AimCallback
from llama_index import SummaryIndex, ServiceContext, SimpleDirectoryReader

# Let's read the documents using `SimpleDirectoryReader` from 'examples/data/paul_graham'.

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

docs = SimpleDirectoryReader("./data/paul_graham").load_data()

# Now lets initialize an AimCallback instance, and add it to the list of callback managers. 

aim_callback = AimCallback(repo="./")
callback_manager = CallbackManager([aim_callback])

# Next, we create an instance of `SummaryIndex` class, by passing in the document reader and the service context. After which we create a query engine which we will use to run queries on the index and retrieve relevant results.

service_context = ServiceContext.from_defaults(
    callback_manager=callback_manager
)
index = SummaryIndex.from_documents(docs, service_context=service_context)
query_engine = index.as_query_engine()

# Finally let's ask a question to the LM based on our provided document

response = query_engine.query("What did the author do growing up?")

# The callback manager will log the `CBEventType.LLM` type of events as an Aim.Text, and we can explore the LM given prompt and the output in the Text Explorer. By first doing `aim up` and navigating by the given url.
