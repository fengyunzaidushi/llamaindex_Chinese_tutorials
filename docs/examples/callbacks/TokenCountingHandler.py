#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/callbacks/TokenCountingHandler.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Token Counting Handler
# 
# This notebook walks through how to use the TokenCountingHandler and how it can be used to track your prompt, completion, and embedding token usage over time.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import tiktoken
from llama_index.llms import Anthropic

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    set_global_service_context,
)
from llama_index.callbacks import CallbackManager, TokenCountingHandler

import os

os.environ["ANTHROPIC_API_KEY"] = "YOUR_API_KEY"

# ## Setup
# 
# Here, we setup the callback and the serivce context. We set a global service context so that we don't have to worry about passing it into indexes and queries.

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

callback_manager = CallbackManager([token_counter])

llm = Anthropic()

service_context = ServiceContext.from_defaults(
    llm=llm, callback_manager=callback_manager, embed_model="local"
)

# set the global default!
set_global_service_context(service_context)

# ## Token Counting
# 
# The token counter will track embedding, prompt, and completion token usage. The token counts are __cummulative__ and are only reset when you choose to do so, with `token_counter.reset_counts()`.
# 
# ### Embedding Token Usage
# 
# Now that the service context is setup, let's track our embedding token usage.

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

documents = SimpleDirectoryReader("./data/paul_graham").load_data()

index = VectorStoreIndex.from_documents(documents)

print(token_counter.total_embedding_token_count)

# That looks right! Before we go any further, lets reset the counts

token_counter.reset_counts()

# ### LLM + Embedding Token Usage
# 
# Next, let's test a query and see what the counts look like.

query_engine = index.as_query_engine(similarity_top_k=4)
response = query_engine.query("What did the author do growing up?")

print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)

# ### Token Counting + Streaming!
# 
# The token counting handler also handles token counting during streaming.
# 
# Here, token counting will only happen once the stream is completed.

token_counter.reset_counts()

query_engine = index.as_query_engine(similarity_top_k=4, streaming=True)
response = query_engine.query("What happened at Interleaf?")

# finish the stream
for token in response.response_gen:
    # print(token, end="", flush=True)
    continue

print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)

# ## Advanced Usage
# 
# The token counter tracks each token usage event in an object called a `TokenCountingEvent`. This object has the following attributes:
# 
# - prompt -> The prompt string sent to the LLM or Embedding model
# - prompt_token_count -> The token count of the LLM prompt
# - completion -> The string completion received from the LLM (not used for embeddings)
# - completion_token_count -> The token count of the LLM completion (not used for embeddings)
# - total_token_count -> The total prompt + completion tokens for the event
# - event_id -> A string ID for the event, which aligns with other callback handlers
# 
# These events are tracked on the token counter in two lists:
# 
# - llm_token_counts
# - embedding_token_counts
# 
# Let's explore what these look like!

print("Num LLM token count events: ", len(token_counter.llm_token_counts))
print(
    "Num Embedding token count events: ",
    len(token_counter.embedding_token_counts),
)

# This makes sense! The previous query embedded the query text, and then made 2 LLM calls (since the top k was 4, and the default chunk size is 1024, two seperate calls need to be made so the LLM can read all the retrieved text).
# 
# Next, let's quickly see what these events look like for a single event.

print("prompt: ", token_counter.llm_token_counts[0].prompt[:100], "...\n")
print(
    "prompt token count: ",
    token_counter.llm_token_counts[0].prompt_token_count,
    "\n",
)

print(
    "completion: ", token_counter.llm_token_counts[0].completion[:100], "...\n"
)
print(
    "completion token count: ",
    token_counter.llm_token_counts[0].completion_token_count,
    "\n",
)

print("total token count", token_counter.llm_token_counts[0].total_token_count)

