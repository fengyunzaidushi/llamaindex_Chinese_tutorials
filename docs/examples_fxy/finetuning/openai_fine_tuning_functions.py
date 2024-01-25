#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/finetuning/openai_fine_tuning_functions.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Fine Tuning with Function Calling
# 

# 
# We will walk through some examples, from simple to advanced:
# 1. Fine-tuning on some toy messages/structured outputs logged through our OpenAI Pydantic Program object.
# 2. Fine-tuning on context-augmented queries/structured outputs over an entire document corpus. Use this in a RAG system.

import nest_asyncio

nest_asyncio.apply()

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

# ## Fine-tuning Using GPT-4 Pydantic Programs
# 

# ### Defining Pydantic Model + Program
# 
# Here, we define the GPT-4 powered function calling program that will generate structured outputs into a Pydantic object (an Album).

from llama_index.program import OpenAIPydanticProgram
from pydantic import BaseModel
from llama_index.llms import OpenAI
from llama_index.callbacks import OpenAIFineTuningHandler
from llama_index.callbacks import CallbackManager
from typing import List

class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int

class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]

finetuning_handler = OpenAIFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])

llm = OpenAI(model="gpt-4", callback_manager=callback_manager)

prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""
program = OpenAIPydanticProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    llm=llm,
    verbose=False,
)

# ### Log Inputs/Outputs
# 
# We define some sample movie names as inputs and log the outputs through the function calling program.

# NOTE: we need >= 10 movies to use OpenAI fine-tuning
movie_names = [
    "The Shining",
    "The Departed",
    "Titanic",
    "Goodfellas",
    "Pretty Woman",
    "Home Alone",
    "Caged Fury",
    "Edward Scissorhands",
    "Total Recall",
    "Ghost",
    "Tremors",
    "RoboCop",
    "Rocky V",
]

from tqdm.notebook import tqdm

for movie_name in tqdm(movie_names):
    output = program(movie_name=movie_name)
    print(output.json())

finetuning_handler.save_finetuning_events("mock_finetune_songs.jsonl")

#('cat mock_finetune_songs.jsonl')

# ### Fine-tune on the Dataset
# 
# We now define a fine-tuning engine and fine-tune on the mock dataset.

from llama_index.finetuning import OpenAIFinetuneEngine

finetune_engine = OpenAIFinetuneEngine(
    "gpt-3.5-turbo",
    "mock_finetune_songs.jsonl",
    # start_job_id="<start-job-id>"  # if you have an existing job, can specify id here
    validate_json=False,  # openai validate json code doesn't support function calling yet
)

finetune_engine.finetune()

finetune_engine.get_current_job()

# ### Try it Out! 
# 
# We obtain the fine-tuned LLM and use it with the Pydantic program.

ft_llm = finetune_engine.get_finetuned_model(temperature=0.3)

ft_program = OpenAIPydanticProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    llm=ft_llm,
    verbose=False,
)

ft_program(movie_name="Goodfellas")

# ## Fine-tuning Structured Outputs through a RAG System
# 
# A use case of function calling is to get structured outputs through a RAG system.
# 
# Here we show how to create a training dataset of context-augmented inputs + structured outputs over an unstructured document. We can then fine-tune the LLM and plug it into a RAG system to perform retrieval + output extraction.

#('mkdir data && wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"')

from pydantic import Field
from typing import List

class Citation(BaseModel):
    """Citation class."""

    author: str = Field(
        ..., description="Inferred first author (usually last name"
    )
    year: int = Field(..., description="Inferred year")
    desc: str = Field(
        ...,
        description=(
            "Inferred description from the text of the work that the author is"
            " cited for"
        ),
    )

class Response(BaseModel):
    """List of author citations.

    Extracted over unstructured text.

    """

    citations: List[Citation] = Field(
        ...,
        description=(
            "List of author citations (organized by author, year, and"
            " description)."
        ),
    )

# ### Load Data + Setup

from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index import Document, ServiceContext
from llama_index.node_parser import SentenceSplitter
from pathlib import Path

loader = PyMuPDFReader()
docs0 = loader.load(file_path=Path("./data/llama2.pdf"))

doc_text = "\n\n".join([d.get_content() for d in docs0])
metadata = {
    "paper_title": "Llama 2: Open Foundation and Fine-Tuned Chat Models"
}
docs = [Document(text=doc_text, metadata=metadata)]

chunk_size = 1024
node_parser = SentenceSplitter(chunk_size=chunk_size)
nodes = node_parser.get_nodes_from_documents(docs)

len(nodes)

# setup service context
finetuning_handler = OpenAIFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])
gpt_4_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4-0613", temperature=0.3),
    callback_manager=callback_manager,
    chunk_size=chunk_size,
)
gpt_35_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo-0613", temperature=0.3),
    callback_manager=callback_manager,
    chunk_size=chunk_size,
)
eval_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4-0613", temperature=0), chunk_size=chunk_size
)

# ### Generate Dataset
# 
# Here we show how to generate a training dataset over these unstructured chunks/nodes.
# 
# We generate questions to extract citations over different context. We run these questions through a GPT-4 RAG pipeline, extract structured outputs, and log inputs/outputs.

# setup dataset generator
from llama_index.evaluation import DatasetGenerator
from llama_index import SummaryIndex, PromptTemplate
from tqdm.notebook import tqdm
from tqdm.asyncio import tqdm_asyncio

fp = open("data/qa_pairs.jsonl", "w")

question_gen_prompt = PromptTemplate(
    """
{query_str}

Context:
{context_str}

Questions:
"""
)

question_gen_query = """\
Snippets from a research paper is given below. It contains citations.
Please generate questions from the text asking about these citations.

For instance, here are some sample questions:
Which citations correspond to related works on transformer models? 
Tell me about authors that worked on advancing RLHF.
Can you tell me citations corresponding to all computer vision works? \
"""

qr_pairs = []
node_questions_tasks = []
for idx, node in enumerate(nodes[:39]):
    num_questions = 1  # change this number to increase number of nodes
    dataset_generator = DatasetGenerator(
        [node],
        question_gen_query=question_gen_query,
        text_question_template=question_gen_prompt,
        service_context=eval_context,
        metadata_mode="all",
        num_questions_per_chunk=num_questions,
    )

    task = dataset_generator.agenerate_questions_from_nodes(num=num_questions)
    node_questions_tasks.append(task)
node_questions_lists = await tqdm_asyncio.gather(*node_questions_tasks)

node_questions_lists

gpt4_index = VectorStoreIndex(nodes, service_context=gpt_4_context)
gpt4_query_engine = gpt4_index.as_query_engine(
    output_cls=Response, similarity_top_k=1
)

from json import JSONDecodeError

for idx, node in enumerate(tqdm(nodes[:39])):
    node_questions_0 = node_questions_lists[idx]
    for question in node_questions_0:
        try:
            # note: we don't need to use response, events are logged through fine-tuning handler
            gpt4_query_engine.query(question)
        except Exception as e:
            print(f"Error for question {question}, {repr(e)}")
            pass

finetuning_handler.save_finetuning_events("llama2_citation_events.jsonl")

# ### Setup Fine-tuning
# 
# We kick off fine-tuning over the generated dataset.

from llama_index.finetuning import OpenAIFinetuneEngine

finetune_engine = OpenAIFinetuneEngine(
    "gpt-3.5-turbo",
    "llama2_citation_events.jsonl",
    # start_job_id="<start-job-id>"  # if you have an existing job, can specify id here
    validate_json=False,  # openai validate json code doesn't support function calling yet
)

finetune_engine.finetune()

finetune_engine.get_current_job()

# ### Use within RAG Pipeline
# 
# Let's plug the fine-tuned LLM into a full RAG pipeline that outputs structured outputs.

ft_llm = finetune_engine.get_finetuned_model(temperature=0.3)
ft_service_context = ServiceContext.from_defaults(llm=ft_llm)

from llama_index import VectorStoreIndex

vector_index = VectorStoreIndex(nodes, service_context=ft_service_context)
query_engine = vector_index.as_query_engine(
    output_cls=Response, similarity_top_k=1
)

# setup baseline as well
base_index = VectorStoreIndex(nodes, service_context=gpt_35_context)
base_query_engine = base_index.as_query_engine(
    output_cls=Response, similarity_top_k=1
)

query_str = """\
Which citation is used to measure the truthfulness of Llama 2? \
"""
# query_str = """\
# Which citation corresponds to the concept of collecting data that represents \
# empirically sampled human preferences in RLHF?\
# """
# query_str = "Which citations in the paper discuss the development and release of Llama 2?"
# query_str = "Which citations are mentioned in the section on RLHF Results?"
# query_str = "Which citation discusses the carbon output related to the production of AI hardware?"

response = query_engine.query(query_str)
print(str(response))

base_response = base_query_engine.query(query_str)
print(str(base_response))

# view sources
print(response.source_nodes[0].get_content())

# as a reference, take a look at GPT-4 response
gpt4_response = gpt4_query_engine.query(query_str)
print(str(gpt4_response))

