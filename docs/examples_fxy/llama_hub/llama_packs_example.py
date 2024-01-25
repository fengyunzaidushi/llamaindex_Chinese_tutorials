#!/usr/bin/env python
# coding: utf-8

# # Llama Packs Example
# 
# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llama_hub/llama_packs_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# This example shows you how to use a simple Llama Pack with VoyageAI. We show the following:
# - How to download a Llama Pack
# - How to inspect its modules
# - How to run it out of the box
# - How to customize it.
# 
# You can find all packs on https://llamahub.ai
# 
# 
# 

# ### Setup Data

#('wget "https://www.dropbox.com/s/f6bmb19xdg0xedm/paul_graham_essay.txt?dl=1" -O paul_graham_essay.txt')

from llama_index import SimpleDirectoryReader

# load in some sample data
reader = SimpleDirectoryReader(input_files=["paul_graham_essay.txt"])
documents = reader.load_data()

# ### Download and Initialize Pack
# 
# We use `download_llama_pack` to download the pack class, and then we initialize it with documents.
# 
# Every pack will have different initialization parameters. You can find more about the initialization parameters for each pack through its [README](https://github.com/logan-markewich/llama-hub/tree/main/llama_hub/llama_packs/voyage_query_engine) (also on LlamaHub).
# 
# **NOTE**: You must also specify an output directory. In this case the pack is downloaded to `voyage_pack`. This allows you to customize and make changes to the file, and import it later! 

from llama_index.llama_pack import download_llama_pack

VoyageQueryEnginePack = download_llama_pack(
    "VoyageQueryEnginePack", "./voyage_pack"
)

voyage_pack = VoyageQueryEnginePack(documents)

# ##

modules = voyage_pack.get_modules()
#display(modules)

llm = modules["llm"]
vector_index = modules["index"]

# try out LLM
response = llm.complete("hello world")
print(str(response))

# try out retriever
retriever = vector_index.as_retriever()
results = retriever.retrieve("What did the author do growing up?")
print(str(results[0].get_content()))

# ### Run Pack
# 
# Every pack has a `run` function that will accomplish a certain task out of the box. Here we will go through the full RAG pipeline with VoyageAI embeddings.

# this will run the full pack
response = voyage_pack.run(
    "What did the author do growing up?", similarity_top_k=2
)

print(str(response))

# ### Try Customizing Pack
# 
# A major feature of LlamaPacks is that you can and should inspect and modify the code templates!
# 

# 
# Let's go into `voyage_pack` and create a copy.
# 
# 1. For demo purposes we'll copy `voyage_pack` into `voyage_pack_copy`.
# 2. Go into `voyage_pack_copy/base.py` and look at the `VoyageQueryEnginePack` class definition. This is where all the core logic lives. As you can see the pack class itself is a very light base abstraction. You're free to copy/paste the code as you wish.
# 3. Go into the line in the `__init__` where it do `llm = OpenAI(model="gpt-4")` and instead change it to `llm = Anthropic()` (which defaults to claude-2).
# 4. Do `from llama_index.llms import Anthropic` and ensure that `ANTHROPIC_API_KEY` is set in your env variable.
# 5. Now you can use!
# 

from voyage_pack_copy.base import VoyageQueryEnginePack

voyage_pack = VoyageQueryEnginePack(documents)

response = voyage_pack.run("What did the author do during his time in RISD?")
print(str(response))

