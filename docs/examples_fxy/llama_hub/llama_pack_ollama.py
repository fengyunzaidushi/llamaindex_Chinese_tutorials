#!/usr/bin/env python
# coding: utf-8

# # Ollama Llama Pack Example

# ### Setup Data

#('wget "https://www.dropbox.com/s/f6bmb19xdg0xedm/paul_graham_essay.txt?dl=1" -O paul_graham_essay.txt')

from llama_index import SimpleDirectoryReader

# load in some sample data
reader = SimpleDirectoryReader(input_files=["paul_graham_essay.txt"])
documents = reader.load_data()

# ### Start Ollama
# 
# Make sure you run `ollama run llama2` in a terminal.

# !ollama run llama2

# ### Download and Initialize Pack
# 
# We use `download_llama_pack` to download the pack class, and then we initialize it with documents.
# 
# Every pack will have different initialization parameters. You can find more about the initialization parameters for each pack through its [README](https://github.com/logan-markewich/llama-hub/tree/main/llama_hub/llama_packs/voyage_query_engine) (also on LlamaHub).
# 
# **NOTE**: You must also specify an output directory. In this case the pack is downloaded to `voyage_pack`. This allows you to customize and make changes to the file, and import it later! 

from llama_index.llama_pack import download_llama_pack

# download and install dependencies
OllamaQueryEnginePack = download_llama_pack(
    "OllamaQueryEnginePack", "./ollama_pack"
)

from ollama_pack.base import OllamaQueryEnginePack

# You can use any llama-hub loader to get documents!
ollama_pack = OllamaQueryEnginePack(model="llama2", documents=documents)

response = ollama_pack.run("What did the author do growing up?")

print(str(response))

