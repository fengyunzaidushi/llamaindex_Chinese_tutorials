#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/embeddings/custom_embeddings.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Custom Embeddings
# LlamaIndex supports embeddings from OpenAI, Azure, and Langchain. But if this isn't enough, you can also implement any embeddings model!
# 
# The example below uses Instructor Embeddings ([install/setup details here](https://huggingface.co/hkunlp/instructor-large)), and implements a custom embeddings class. Instructor embeddings work by providing text, as well as "instructions" on the domain of the text to embed. This is helpful when embedding text from a very specific and specialized topic.
# 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# !pip install InstructorEmbedding torch transformers sentence-transformers

import openai
import os

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
openai.api_key = os.environ["OPENAI_API_KEY"]

# ## Custom Embeddings Implementation

from typing import Any, List
from InstructorEmbedding import INSTRUCTOR

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.embeddings.base import BaseEmbedding

class InstructorEmbeddings(BaseEmbedding):
    _model: INSTRUCTOR = PrivateAttr()
    _instruction: str = PrivateAttr()

    def __init__(
        self,
        instructor_model_name: str = "hkunlp/instructor-large",
        instruction: str = "Represent a document for semantic search:",
        **kwargs: Any,
    ) -> None:
        self._model = INSTRUCTOR(instructor_model_name)
        self._instruction = instruction
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            [[self._instruction, text] for text in texts]
        )
        return embeddings

# ## Usage Example

from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Load Documents

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

service_context = ServiceContext.from_defaults(
    embed_model=InstructorEmbeddings(embed_batch_size=2), chunk_size=512
)

# if running for the first time, will download model weights first!
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

response = index.as_query_engine().query("What did the author do growing up?")
print(response)

