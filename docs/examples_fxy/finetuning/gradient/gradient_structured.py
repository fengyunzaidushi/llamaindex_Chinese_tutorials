#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/finetuning/gradient/gradient_structured.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Fine Tuning Llama2 for Better Structured Outputs With Gradient and LlamaIndex
# 

# 
# We do this by using [gradient.ai](https://gradient.ai)
# 
# This is similar in format to our [OpenAI Functions Fine-tuning Notebook](https://docs.llamaindex.ai/en/latest/examples/finetuning/openai_fine_tuning_functions.html).
# 
# **NOTE**: This is an alternative to our repo/guide on fine-tuning llama2-7b with Modal: https://github.com/run-llama/modal_finetune_sql

#('pip install llama-index gradientai -q')

import os
from llama_index.llms import GradientBaseModelLLM
from llama_index.finetuning.gradient.base import GradientFinetuneEngine

os.environ["GRADIENT_ACCESS_TOKEN"] = os.getenv("GRADIENT_API_KEY")
os.environ["GRADIENT_WORKSPACE_ID"] = "<insert_workspace_id>"

# ## Fine-tuning Using GPT-4 Pydantic Programs
# 

from pydantic import BaseModel

class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str

from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import OpenAI, GradientBaseModelLLM
from llama_index.program import LLMTextCompletionProgram
from llama_index.output_parsers import PydanticOutputParser

openai_handler = LlamaDebugHandler()
openai_callback = CallbackManager([openai_handler])
openai_llm = OpenAI(model="gpt-4", callback_manager=openai_callback)

gradient_handler = LlamaDebugHandler()
gradient_callback = CallbackManager([gradient_handler])
base_model_slug = "llama2-7b-chat"
gradient_llm = GradientBaseModelLLM(
    base_model_slug=base_model_slug,
    max_tokens=300,
    callback_manager=gradient_callback,
    is_chat_model=True,
)
# HACK: set chat model
# from llama_index.llms.types import LLMMetadata
# gradient_llm.metadata = LLMMetadata(
#     context_window=1024,
#     num_output=gradient_llm.max_tokens or 20,
#     is_chat_model=True,
#     is_function_calling_model=False,
#     model_name=gradient_llm._model.id,
# )

# try running both through LLMTextCompletionProgram

prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""
openai_program = LLMTextCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(Album),
    prompt_template_str=prompt_template_str,
    llm=openai_llm,
    verbose=True,
)
gradient_program = LLMTextCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(Album),
    prompt_template_str=prompt_template_str,
    llm=gradient_llm,
    verbose=True,
)

response = openai_program(movie_name="The Shining")
print(str(response))

tmp = openai_handler.get_llm_inputs_outputs()
print(tmp[0][0].payload["messages"][0])

# print(tmp[0][1])

response = gradient_program(movie_name="The Shining")
print(str(response))

tmp = gradient_handler.get_llm_inputs_outputs()
print(tmp[0][0].payload["messages"][0])

# ### Defining Pydantic Model + Program
# 
# Here, we define the GPT-4 powered function calling program that will generate structured outputs into a Pydantic object (an Album).

from llama_index.program import LLMTextCompletionProgram
from pydantic import BaseModel
from llama_index.llms import OpenAI
from llama_index.callbacks import GradientAIFineTuningHandler
from llama_index.callbacks import CallbackManager
from llama_index.output_parsers import PydanticOutputParser
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

finetuning_handler = GradientAIFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])

llm_gpt4 = OpenAI(model="gpt-4", callback_manager=callback_manager)

prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""
openai_program = LLMTextCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(Album),
    prompt_template_str=prompt_template_str,
    llm=llm_gpt4,
    verbose=True,
)

# ### Log Inputs/Outputs
# 
# We define some sample movie names as inputs and log the outputs through the function calling program.

# NOTE: we need >= 10 movies to use Gradient fine-tuning
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
    output = openai_program(movie_name=movie_name)
    print(output.json())

events = finetuning_handler.get_finetuning_events()

events

finetuning_handler.save_finetuning_events("mock_finetune_songs.jsonl")

#('cat mock_finetune_songs.jsonl')

# ### Fine-tune on the Dataset
# 
# We now define a fine-tuning engine and fine-tune on the mock dataset.

# define base model
base_model_slug = "llama2-7b-chat"
base_llm = GradientBaseModelLLM(
    base_model_slug=base_model_slug, max_tokens=500, is_chat_model=True
)

from llama_index.finetuning import GradientFinetuneEngine

finetune_engine = GradientFinetuneEngine(
    base_model_slug=base_model_slug,
    # model_adapter_id='805c6fd6-daa8-4fc8-a509-bebb2f2c1024_model_adapter',
    name="movies_structured",
    data_path="mock_finetune_songs.jsonl",
    verbose=True,
    max_steps=200,
    batch_size=1,
)

finetune_engine.model_adapter_id

# asdjust epochs as necessary
epochs = 2
for i in range(epochs):
    print(f"** EPOCH {i} **")
    finetune_engine.finetune()

ft_llm = finetune_engine.get_finetuned_model(
    max_tokens=500, is_chat_model=True
)

# # NOTE: same as doing the following
# from llama_index.llms import GradientModelAdapterLLM

# ft_llm = GradientModelAdapterLLM(
#     model_adapter_id=finetune_engine.model_adapter_id,
#     max_tokens=500
# )

# ### Try it Out! 
# 
# We obtain the fine-tuned LLM and use it with the Pydantic program.

# try a slightly modified prompt_template_str
new_prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\

Please only generate one album.
"""

gradient_program = LLMTextCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(Album),
    # prompt_template_str=prompt_template_str,
    prompt_template_str=new_prompt_template_str,
    llm=ft_llm,
    verbose=True,
)

gradient_program(movie_name="Goodfellas")

gradient_program(movie_name="Chucky")

# you wouldn't get this with normal llama2-7b!
base_gradient_program = LLMTextCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(Album),
    prompt_template_str=prompt_template_str,
    llm=base_llm,
    verbose=True,
)

# throws an error
base_gradient_program(movie_name="Goodfellas")

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
from llama_index.node_parser import SimpleNodeParser
from pathlib import Path
from llama_index.callbacks import GradientAIFineTuningHandler

loader = PyMuPDFReader()
docs0 = loader.load(file_path=Path("./data/llama2.pdf"))

doc_text = "\n\n".join([d.get_content() for d in docs0])
metadata = {
    "paper_title": "Llama 2: Open Foundation and Fine-Tuned Chat Models"
}
docs = [Document(text=doc_text, metadata=metadata)]

chunk_size = 1024
node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
nodes = node_parser.get_nodes_from_documents(docs)

len(nodes)

# setup GPT-4 context - to generate "ground-truth" data given queries
finetuning_handler = GradientAIFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])
gpt_4_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4-0613", temperature=0.3),
    callback_manager=callback_manager,
    chunk_size=chunk_size,
    # force using prompts instead of openai function calling for structured outputs
    pydantic_program_mode="llm",
)

# setup gradient.ai context
base_model_slug = "llama2-7b-chat"
base_llm = GradientBaseModelLLM(
    base_model_slug=base_model_slug, max_tokens=500, is_chat_model=True
)
gradient_context = ServiceContext.from_defaults(
    llm=base_llm,
    # callback_manager=callback_manager,
    chunk_size=chunk_size,
    pydantic_program_mode="llm",
)

# setup eval context (for question generation)
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

len(node_questions_lists)

node_questions_lists[1]

# [optional] save
import pickle

pickle.dump(node_questions_lists, open("llama2_questions.pkl", "wb"))

# [optional] load questions
node_questions_lists = pickle.load(open("llama2_questions.pkl", "rb"))

from llama_index import VectorStoreIndex

gpt4_index = VectorStoreIndex(nodes[:39], service_context=gpt_4_context)
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

from llama_index.finetuning import GradientFinetuneEngine

finetune_engine = GradientFinetuneEngine(
    base_model_slug=base_model_slug,
    # model_adapter_id='23a71710-47b3-43be-9be2-58a3efbccf2b_model_adapter',
    name="llama2_structured",
    data_path="llama2_citation_events.jsonl",
    verbose=True,
    max_steps=200,
    batch_size=1,
)

# save this for future runs
finetune_engine.model_adapter_id

# asdjust epochs as necessary
epochs = 2
for i in range(epochs):
    print(f"** EPOCH {i} **")
    finetune_engine.finetune()

# ### Use within RAG Pipeline
# 
# Let's plug the fine-tuned LLM into a full RAG pipeline that outputs structured outputs.

ft_llm = finetune_engine.get_finetuned_model(max_tokens=500)
ft_service_context = ServiceContext.from_defaults(llm=ft_llm)

from llama_index import VectorStoreIndex

vector_index = VectorStoreIndex(nodes, service_context=ft_service_context)
query_engine = vector_index.as_query_engine(
    output_cls=Response, similarity_top_k=1
)

# setup baseline as well
base_index = VectorStoreIndex(nodes, service_context=gradient_context)
base_query_engine = base_index.as_query_engine(
    output_cls=Response, similarity_top_k=1
)

query_str = "Which citations are mentioned in the section about RLHF Results?"
# query_str = """\
# Which citation corresponds to the concept of collecting data that represents \
# empirically sampled human preferences in RLHF?\
# """
# query_str = "Which citations in the paper discuss the development and release of Llama 2?"
# query_str = "Which citations are mentioned in the section on RLHF Results?"
# query_str = "Which citation discusses the carbon output related to the production of AI hardware?"

response = query_engine.query(query_str)
print(str(response))

# Let's take a look at sources

# view sources
print(response.source_nodes[0].get_content())

# Let's compare against the baseline (the base llama2-7b model). Notice that the query engine throws an error! 

# throws an error!
base_response = base_query_engine.query(query_str)
print(str(base_response))

# As a reference, let's also compare against gpt-4.

# as a reference, take a look at GPT-4 response
gpt4_response = gpt4_query_engine.query(query_str)
print(str(gpt4_response))
