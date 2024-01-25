#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/llama_2_llama_cpp.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # LlamaCPP 
# 

# 

# 
# Note that if you're using a version of `llama-cpp-python` after version `0.1.79`, the model format has changed from `ggmlv3` to `gguf`. Old model files like the used in this notebook can be converted using scripts in the [`llama.cpp`](https://github.com/ggerganov/llama.cpp) repo. Alternatively, you can download the GGUF version of the model above from [huggingface](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF).
# 
# By default, if model_path and model_url are blank, the `LlamaCPP` module will load llama2-chat-13B in either format depending on your version.
# 
# #
# 
# To get the best performance out of `LlamaCPP`, it is recomended to install the package so that it is compilied with GPU support. A full guide for installing this way is [here](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal).
# 
# Full MACOS instructions are also [here](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/).
# 

# - Use `CuBLAS` if you have CUDA and an NVidia GPU
# - Use `METAL` if you are running on an M1/M2 MacBook
# - Use `CLBLAST` if you are running on an AMD/Intel GPU

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

# ## Setup LLM
# 
# The LlamaCPP llm is highly configurable. Depending on the model being used, you'll want to pass in `messages_to_prompt` and `completion_to_prompt` functions to help format the model inputs.
# 
# Since the default model is llama2-chat, we use the util functions found in [`llama_index.llms.llama_utils`](https://github.com/jerryjliu/llama_index/blob/main/llama_index/llms/llama_utils.py).
# 
# For any kwargs that need to be passed in during initialization, set them in `model_kwargs`. A full list of available model kwargs is available in the [LlamaCPP docs](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.llama.Llama.__init__).
# 
# For any kwargs that need to be passed in during inference, you can set them in `generate_kwargs`. See the full list of [generate kwargs here](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.llama.Llama.__call__).
# 

# 
# As noted above, we're using the [`llama-2-chat-13b-ggml`](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML) model in this notebook which uses the `ggmlv3` model format. If you are running a version of `llama-cpp-python` greater than `0.1.79`, you can replace the `model_url` below with `"https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"`.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

# We can tell that the model is using `metal` due to the logging!

# ## Start using our `LlamaCPP` LLM abstraction!
# 
# We can simply use the `complete` method of our `LlamaCPP` LLM abstraction to generate completions given a prompt.

response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
print(response.text)

# We can use the `stream_complete` endpoint to stream the response as it’s being generated rather than waiting for the entire response to be generated.

response_iter = llm.stream_complete("Can you write me a poem about fast cars?")
for response in response_iter:
    print(response.delta, end="", flush=True)

# ## Query engine set up with LlamaCPP
# 
# We can simply pass in the `LlamaCPP` LLM abstraction to the `LlamaIndex` query engine as usual.
# 
# But first, let's change the global tokenizer to match our LLM.

from llama_index import set_global_tokenizer
from transformers import AutoTokenizer

set_global_tokenizer(
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
)

# use Huggingface embeddings
from llama_index.embeddings import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# create a service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

# load documents
documents = SimpleDirectoryReader(
    "../../../examples/paul_graham_essay/data"
).load_data()

# create vector store index
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# set up query engine
query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")
print(response)

