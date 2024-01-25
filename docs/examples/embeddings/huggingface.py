#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/embeddings/huggingface.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Local Embeddings with HuggingFace
# 
# LlamaIndex has support for HuggingFace embedding models, including BGE, Instructor, and more.
# 
# Furthermore, we provide utilties to create and use ONNX models using the [Optimum library](https://huggingface.co/docs/transformers/serialization#exporting-a-transformers-model-to-onnx-with-optimumonnxruntime) from HuggingFace.

# ## HuggingFaceEmbedding
# 
# The base `HuggingFaceEmbedding` class is a generic wrapper around any HuggingFace model for embeddings. You can set either `pooling="cls"` or `pooling="mean"` -- in most cases, you'll want `cls` pooling. But the model card for your particular model may have other recommendations.
# 
# You can refer to the [embeddings leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for more recommendations on embedding models.
# 
# This class depends on the transformers package, which you can install with `pip install transformers`.
# 
# NOTE: if you were previously using a `HuggingFaceEmbeddings` from LangChain, this should give equivilant results.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.embeddings import HuggingFaceEmbedding

# loads BAAI/bge-small-en
# embed_model = HuggingFaceEmbedding()

# loads BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

embeddings = embed_model.get_text_embedding("Hello World!")
print(len(embeddings))
print(embeddings[:5])

# #
# 

# 
# They rely on the `Instructor` pip package, which you can install with `pip install InstructorEmbedding`.

from llama_index.embeddings import InstructorEmbedding

embed_model = InstructorEmbedding(model_name="hkunlp/instructor-base")

embeddings = embed_model.get_text_embedding("Hello World!")
print(len(embeddings))
print(embeddings[:5])

# ## OptimumEmbedding
# 
# Optimum in a HuggingFace library for exporting and running HuggingFace models in the ONNX format.
# 
# You can install the dependencies with `pip install transformers optimum[exporters]`.
# 
# First, we need to create the ONNX model. ONNX models provide improved inference speeds, and can be used across platforms (i.e. in TransformersJS)

from llama_index.embeddings import OptimumEmbedding

OptimumEmbedding.create_and_save_optimum_model(
    "BAAI/bge-small-en-v1.5", "./bge_onnx"
)

embed_model = OptimumEmbedding(folder_name="./bge_onnx")

embeddings = embed_model.get_text_embedding("Hello World!")
print(len(embeddings))
print(embeddings[:5])

# ## Benchmarking
# 
# Let's try comparing using a classic large document -- the IPCC climate report, chapter 3.

#('curl https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter03.pdf --output IPCC_AR6_WGII_Chapter03.pdf')

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

documents = SimpleDirectoryReader(
    input_files=["IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# ### Base HuggingFace Embeddings

import os
import openai

# needed to synthesize responses later
os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index.embeddings import HuggingFaceEmbedding

# loads BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
test_emeds = embed_model.get_text_embedding("Hello World!")

service_context = ServiceContext.from_defaults(embed_model=embed_model)

get_ipython().run_cell_magic('timeit', '-r 1 -n 1', 'index = VectorStoreIndex.from_documents(\n    documents, service_context=service_context, show_progress=True\n)\n')

# ### Optimum Embeddings
# 
# We can use the onnx embeddings we created earlier

from llama_index.embeddings import OptimumEmbedding

embed_model = OptimumEmbedding(folder_name="./bge_onnx")
test_emeds = embed_model.get_text_embedding("Hello World!")

service_context = ServiceContext.from_defaults(embed_model=embed_model)

get_ipython().run_cell_magic('timeit', '-r 1 -n 1', 'index = VectorStoreIndex.from_documents(\n    documents, service_context=service_context, show_progress=True\n)\n')

