#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/llama_datasets/labelled-rag-datasets.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Benchmarking RAG Pipelines With A `LabelledRagDatatset`
# 
# The `LabelledRagDataset` is meant to be used for evaluating any given RAG pipeline, for which there could be several configurations (i.e. choosing the `LLM`, values for the `similarity_top_k`, `chunk_size`, and others). We've likened this abstract to traditional machine learning datastets, where `X` features are meant to predict a ground-truth label `y`. In this case, we use the `query` as well as the retrieved `contexts` as the "features" and the answer to the query, called `reference_answer` as the ground-truth label.
# 
# And of course, such datasets are comprised of observations or examples. In the case of `LabelledRagDataset`, these are made up with a set of `LabelledRagDataExample`'s.
# 

# ### The `LabelledRagDataExample` Class

from llama_index.llama_dataset import (
    LabelledRagDataExample,
    CreatedByType,
    CreatedBy,
)

# constructing a LabelledRagDataExample
query = "This is a test query, is it not?"
query_by = CreatedBy(type=CreatedByType.AI, model_name="gpt-4")
reference_answer = "Yes it is."
reference_answer_by = CreatedBy(type=CreatedByType.HUMAN)
reference_contexts = ["This is a sample context"]

rag_example = LabelledRagDataExample(
    query=query,
    query_by=query_by,
    reference_contexts=reference_contexts,
    reference_answer=reference_answer,
    reference_answer_by=reference_answer_by,
)

# The `LabelledRagDataExample` is a Pydantic `Model` and so, going from `json` or `dict` (and vice-versa) is possible.

print(rag_example.json())

LabelledRagDataExample.parse_raw(rag_example.json())

rag_example.dict()

LabelledRagDataExample.parse_obj(rag_example.dict())

# Let's create a second example, so we can have a (slightly) more interesting `LabelledRagDataset`.

query = "This is a test query, is it so?"
reference_answer = "I think yes, it is."
reference_contexts = ["This is a second sample context"]

rag_example_2 = LabelledRagDataExample(
    query=query,
    query_by=query_by,
    reference_contexts=reference_contexts,
    reference_answer=reference_answer,
    reference_answer_by=reference_answer_by,
)

# ### The `LabelledRagDataset` Class

from llama_index.llama_dataset.rag import LabelledRagDataset

rag_dataset = LabelledRagDataset(examples=[rag_example, rag_example_2])

# There exists a convienience method to view the dataset as a `pandas.DataFrame`.

rag_dataset.to_pandas()

# #### Serialization

# To persist and load the dataset to and from disk, there are the `save_json` and `from_json` methods.

rag_dataset.save_json("rag_dataset.json")

reload_rag_dataset = LabelledRagDataset.from_json("rag_dataset.json")

reload_rag_dataset.to_pandas()

# ### Building a synthetic `LabelledRagDataset` over Wikipedia 
# 
# For this section, we'll first create a `LabelledRagDataset` using a synthetic generator. Ultimately, we will use GPT-4 to produce both the `query` and `reference_answer` for the synthetic `LabelledRagDataExample`'s.
# 
# NOTE: if one has queries, reference answers, and contexts over a text corpus, then it is not necessary to use data synthesis to be able to predict and subsequently evaluate said predictions.

import nest_asyncio

nest_asyncio.apply()

#('pip install wikipedia -q')

# wikipedia pages
from llama_index.readers import WikipediaReader
from llama_index import VectorStoreIndex

cities = [
    "San Francisco",
]

documents = WikipediaReader().load_data(
    pages=[f"History of {x}" for x in cities]
)
index = VectorStoreIndex.from_documents(documents)

# The `RagDatasetGenerator` can be built over a set of documents to generate `LabelledRagDataExample`'s.

# generate questions against chunks
from llama_index.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms import OpenAI
from llama_index import ServiceContext

# set context for llm provider
gpt_35_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3)
)

# instantiate a DatasetGenerator
dataset_generator = RagDatasetGenerator.from_documents(
    documents,
    service_context=gpt_35_context,
    num_questions_per_chunk=2,  # set the number of questions per nodes
    show_progress=True,
)

len(dataset_generator.nodes)

# since there are 13 nodes, there should be a total of 26 questions
rag_dataset = dataset_generator.generate_dataset_from_nodes()

rag_dataset.to_pandas()

rag_dataset.save_json("rag_dataset.json")

