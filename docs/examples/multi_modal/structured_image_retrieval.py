#!/usr/bin/env python
# coding: utf-8

# # Semi-structured Image Retrieval
# 

# 
# Given a set of images, we can infer structured outputs from them using Gemini Pro Vision.
# 
# We can then index these structured outputs in a vector database. We then take full advantage of semantic search + metadata filter capabilities with **auto-retrieval**: this allows us to ask both structured and semantic questions over this data!
# 
# (An alternative is to put this data into a SQL database, letting you do text-to-SQL. These techniques are quite closely related).

#("pip install llama-index 'google-generativeai>=0.3.0' matplotlib qdrant_client")

# ## Setup

# ### Get Google API Key

import os

GOOGLE_API_KEY = ""  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ### Download Images
# 
# We download the full SROIE v2 dataset from Kaggle [here](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2).
# 
# This dataset consists of scanned receipt images. We ignore the ground-truth labels for now, and use the test set images to test out Gemini's capabilities for structured output extraction.

# ### Get Image Files
# 
# Now that the images are downloaded, we can get a list of the file names.

from pathlib import Path
import random
from typing import Optional

def get_image_files(
    dir_path, sample: Optional[int] = 10, shuffle: bool = False
):
    dir_path = Path(dir_path)
    image_paths = []
    for image_path in dir_path.glob("*.jpg"):
        image_paths.append(image_path)

    random.shuffle(image_paths)
    if sample:
        return image_paths[:sample]
    else:
        return image_paths

image_files = get_image_files("SROIE2019/test/img", sample=100)

# ## Use Gemini to extract structured outputs
# 
# Here we use Gemini to extract structured outputs.
# 1. Define a ReceiptInfo pydantic class that captures the structured outputs we want to extract. We extract fields like `company`, `date`, `total`, and also `summary`.
# 2. Define a `pydantic_gemini` function which will convert input documents into a response.

# ### Define a ReceiptInfo pydantic class

from pydantic import BaseModel, Field

class ReceiptInfo(BaseModel):
    company: str = Field(..., description="Company name")
    date: str = Field(..., description="Date field in DD/MM/YYYY format")
    address: str = Field(..., description="Address")
    total: float = Field(..., description="total amount")
    currency: str = Field(
        ..., description="Currency of the country (in abbreviations)"
    )
    summary: str = Field(
        ...,
        description="Extracted text summary of the receipt, including items purchased, the type of store, the location, and any other notable salient features (what does the purchase seem to be for?).",
    )

# ### Define a `pydantic_gemini` function

from llama_index.multi_modal_llms import GeminiMultiModal
from llama_index.program import MultiModalLLMCompletionProgram
from llama_index.output_parsers import PydanticOutputParser

prompt_template_str = """\
    Can you summarize the image and return a response \
    with the following JSON format: \
"""

async def pydantic_gemini(output_class, image_documents, prompt_template_str):
    gemini_llm = GeminiMultiModal(
        api_key=GOOGLE_API_KEY, model_name="models/gemini-pro-vision"
    )

    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_class),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=gemini_llm,
        verbose=True,
    )

    response = await llm_program.acall()
    return response

# ### Run over images

from llama_index import SimpleDirectoryReader
from llama_index.async_utils import run_jobs

async def aprocess_image_file(image_file):
    # should load one file
    print(f"Image file: {image_file}")
    img_docs = SimpleDirectoryReader(input_files=[image_file]).load_data()
    output = await pydantic_gemini(ReceiptInfo, img_docs, prompt_template_str)
    return output

async def aprocess_image_files(image_files):
    """Process metadata on image files."""

    new_docs = []
    tasks = []
    for image_file in image_files:
        task = aprocess_image_file(image_file)
        tasks.append(task)

    outputs = await run_jobs(tasks, show_progress=True, workers=5)
    return outputs

outputs = await aprocess_image_files(image_files)

outputs[4]

# ### Convert Structured Representation to `TextNode` objects
# 
# Node objects are the core units that are indexed in vector stores in LlamaIndex. We define a simple converter function to map the `ReceiptInfo` objects to `TextNode` objects.

from llama_index.schema import TextNode
from typing import List

def get_nodes_from_objs(
    objs: List[ReceiptInfo], image_files: List[str]
) -> TextNode:
    """Get nodes from objects."""
    nodes = []
    for image_file, obj in zip(image_files, objs):
        node = TextNode(
            text=obj.summary,
            metadata={
                "company": obj.company,
                "date": obj.date,
                "address": obj.address,
                "total": obj.total,
                "currency": obj.currency,
                "image_file": str(image_file),
            },
            excluded_embed_metadata_keys=["image_file"],
            excluded_llm_metadata_keys=["image_file"],
        )
        nodes.append(node)
    return nodes

nodes = get_nodes_from_objs(outputs, image_files)

print(nodes[0].get_content(metadata_mode="all"))

# ##

import qdrant_client
from llama_index.vector_stores import QdrantVectorStore
from llama_index.storage import StorageContext
from llama_index import VectorStoreIndex

# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_gemini")

vector_store = QdrantVectorStore(client=client, collection_name="collection")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)

# ## Define Auto-Retriever
# 
# Now we can setup our auto-retriever, which can perform semi-structured queries: structured queries through inferring metadata filters, along with semantic search.
# 
# We setup our schema definition capturing the receipt info which is fed into the prompt.

from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo

vector_store_info = VectorStoreInfo(
    content_info="Receipts",
    metadata_info=[
        MetadataInfo(
            name="company",
            description="The name of the store",
            type="string",
        ),
        MetadataInfo(
            name="address",
            description="The address of the store",
            type="string",
        ),
        MetadataInfo(
            name="date",
            description="The date of the purchase (in DD/MM/YYYY format)",
            type="string",
        ),
        MetadataInfo(
            name="total",
            description="The final amount",
            type="float",
        ),
        MetadataInfo(
            name="currency",
            description="The currency of the country the purchase was made (abbreviation)",
            type="string",
        ),
    ],
)

from llama_index.retrievers import VectorIndexAutoRetriever

retriever = VectorIndexAutoRetriever(
    index,
    vector_store_info=vector_store_info,
    similarity_top_k=2,
    empty_query_top_k=10,  # if only metadata filters are specified, this is the limit
    verbose=True,
)

# from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from IPython.#display import Image

def #display_response(nodes: List[TextNode]):
    """Display response."""
    for node in nodes:
        print(node.get_content(metadata_mode="all"))
        # img = Image.open(open(node.metadata["image_file"], 'rb'))
        #display(Image(filename=node.metadata["image_file"], width=200))

# ## Run Some Queries
# 
# Let's try out different types of queries!

nodes = retriever.retrieve(
    "Tell me about some restaurant orders of noodles with total < 25"
)
#display_response(nodes)

nodes = retriever.retrieve("Tell me about some grocery purchases")
#display_response(nodes)

