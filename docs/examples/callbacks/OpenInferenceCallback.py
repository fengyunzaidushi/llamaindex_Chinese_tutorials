#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/callbacks/OpenInferenceCallback.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # OpenInference Callback Handler + Arize Phoenix
# 
# [OpenInference](https://github.com/Arize-ai/open-inference-spec) is an open standard for capturing and storing AI model inferences. It enables production LLMapp servers to seamlessly integrate with LLM observability solutions such as [Arize](https://arize.com/) and [Phoenix](https://github.com/Arize-ai/phoenix).
# 
# The `OpenInferenceCallbackHandler` saves data from LLM applications for downstream analysis and debugging. In particular, it saves the following data in columnar format:
# 
# - query IDs
# - query text
# - query embeddings
# - scores (e.g., cosine similarity)
# - retrieved document IDs
# 
# This tutorial demonstrates the callback handler's use for both in-notebook experimentation and lightweight production logging.
# 
# ⚠️ The `OpenInferenceCallbackHandler` is in beta and its APIs are subject to change.
# 
# ℹ️ If you find that your particular query engine or use-case is not supported, open an issue on [GitHub](https://github.com/Arize-ai/open-inference-spec/issues).

# #
# 

#('pip install -q html2text llama-index pandas tqdm')

# Import libraries.
# 

import hashlib
import json
from pathlib import Path
import os
import textwrap
from typing import List, Union

from llama_index import (
    SimpleWebPageReader,
    ServiceContext,
    VectorStoreIndex,
)
from llama_index.callbacks import CallbackManager, OpenInferenceCallbackHandler
from llama_index.callbacks.open_inference_callback import (
    as_dataframe,
    QueryData,
    NodeData,
)
from llama_index.node_parser import SimpleNodeParser
import pandas as pd
from tqdm import tqdm

# ## Load and Parse Documents
# 
# Load documents from Paul Graham's essay "What I Worked On".

documents = SimpleWebPageReader().load_data(
    [
        "http://raw.githubusercontent.com/jerryjliu/llama_index/main/examples/paul_graham_essay/data/paul_graham_essay.txt"
    ]
)
print(documents[0].text)

# Parse the document into nodes. Display the first node's text.

parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)
print(nodes[0].text)

# ## Access Data as a Pandas Dataframe
# 
# When experimenting with chatbots and LLMapps in a notebook, it's often useful to run your chatbot against a small collection of user queries and collect and analyze the data for iterative improvement. The `OpenInferenceCallbackHandler` stores your data in columnar format and provides convenient access to the data as a pandas dataframe.
# 

callback_handler = OpenInferenceCallbackHandler()
callback_manager = CallbackManager([callback_handler])
service_context = ServiceContext.from_defaults(
    callback_manager=callback_manager
)

# Build the index and instantiate the query engine.

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
query_engine = index.as_query_engine()

# Run your query engine across a collection of queries.

max_characters_per_line = 80
queries = [
    "What did Paul Graham do growing up?",
    "When and how did Paul Graham's mother die?",
    "What, in Paul Graham's opinion, is the most distinctive thing about YC?",
    "When and how did Paul Graham meet Jessica Livingston?",
    "What is Bel, and when and where was it written?",
]
for query in queries:
    response = query_engine.query(query)
    print("Query")
    print("=====")
    print(textwrap.fill(query, max_characters_per_line))
    print()
    print("Response")
    print("========")
    print(textwrap.fill(str(response), max_characters_per_line))
    print()

# The data from your query engine runs can be accessed as a pandas dataframe for analysis and iterative improvement.

query_data_buffer = callback_handler.flush_query_data_buffer()
query_dataframe = as_dataframe(query_data_buffer)
query_dataframe

# The dataframe column names conform to the OpenInference spec, which specifies the category, data type, and intent of each column.

# ## Log Production Data
# 

class ParquetCallback:
    def __init__(
        self, data_path: Union[str, Path], max_buffer_length: int = 1000
    ):
        self._data_path = Path(data_path)
        self._data_path.mkdir(parents=True, exist_ok=False)
        self._max_buffer_length = max_buffer_length
        self._batch_index = 0

    def __call__(
        self,
        query_data_buffer: List[QueryData],
        node_data_buffer: List[NodeData],
    ) -> None:
        if len(query_data_buffer) > self._max_buffer_length:
            query_dataframe = as_dataframe(query_data_buffer)
            file_path = self._data_path / f"log-{self._batch_index}.parquet"
            query_dataframe.to_parquet(file_path)
            self._batch_index += 1
            query_data_buffer.clear()  # ⚠️ clear the buffer or it will keep growing forever!
            node_data_buffer.clear()  # didn't log node_data_buffer, but still need to clear it

# ⚠️ In a production setting, it's important to clear the buffer, otherwise, the callback handler will indefinitely accumulate data in memory and eventually cause your system to crash.

# Attach the Parquet writer to your callback and re-run the query engine. The data will be saved to disk.

data_path = "data"
parquet_writer = ParquetCallback(
    data_path=data_path,
    # this parameter is set artificially low for demonstration purposes
    # to force a flush to disk, in practice it would be much larger
    max_buffer_length=1,
)
callback_handler = OpenInferenceCallbackHandler(callback=parquet_writer)
callback_manager = CallbackManager([callback_handler])
service_context = ServiceContext.from_defaults(
    callback_manager=callback_manager
)
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
query_engine = index.as_query_engine()

for query in tqdm(queries):
    query_engine.query(query)

# Load and #display saved Parquet data from disk to verify that the logger is working. 

query_dataframes = []
for file_name in os.listdir(data_path):
    file_path = os.path.join(data_path, file_name)
    query_dataframes.append(pd.read_parquet(file_path))
query_dataframe = pd.concat(query_dataframes)
query_dataframe

