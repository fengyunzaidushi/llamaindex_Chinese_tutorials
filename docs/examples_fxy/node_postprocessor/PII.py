#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/node_postprocessor/PII.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # PII Masking

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.postprocessor import (
    PIINodePostprocessor,
    NERPIINodePostprocessor,
)
from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext, Document, VectorStoreIndex
from llama_index.schema import TextNode

# load documents
text = """
Hello Paulo Santos. The latest statement for your credit card account \
1111-0000-1111-0000 was mailed to 123 Any Street, Seattle, WA 98109.
"""
node = TextNode(text=text)

# ### Option 1: Use NER Model for PII Masking
# 
# Use a Hugging Face NER model for PII Masking

service_context = ServiceContext.from_defaults()
processor = NERPIINodePostprocessor(service_context=service_context)

from llama_index.schema import NodeWithScore

new_nodes = processor.postprocess_nodes([NodeWithScore(node=node)])

# view redacted text
new_nodes[0].node.get_text()

# get mapping in metadata
# NOTE: this is not sent to the LLM!
new_nodes[0].node.metadata["__pii_node_info__"]

# ### Option 2: Use LLM for PII Masking
# 
# NOTE: You should be using a *local* LLM model for PII masking. The example shown is using OpenAI, but normally you'd use an LLM running locally, possibly from huggingface. Examples for local LLMs are [here](https://gpt-index.readthedocs.io/en/latest/how_to/customization/custom_llms.html#example-using-a-huggingface-llm).

service_context = ServiceContext.from_defaults()
processor = PIINodePostprocessor(service_context=service_context)

from llama_index.schema import NodeWithScore

new_nodes = processor.postprocess_nodes([NodeWithScore(node=node)])

# view redacted text
new_nodes[0].node.get_text()

# get mapping in metadata
# NOTE: this is not sent to the LLM!
new_nodes[0].node.metadata["__pii_node_info__"]

# ### Feed Nodes to Index

# feed into index
index = VectorStoreIndex([n.node for n in new_nodes])

response = index.as_query_engine().query(
    "What address was the statement mailed to?"
)
print(str(response))

