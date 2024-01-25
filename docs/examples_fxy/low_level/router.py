#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/low_level/router.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 

# # Building a Router from Scratch
# 

# 
# Routers are a simple but effective form of automated decision making that can allow you to perform dynamic retrieval/querying over your data.
# 

# 
# To build a router, we'll walk through the following steps:
# - Crafting an initial prompt to select a set of choices
# - Enforcing structured output (for text completion endpoints)
# - Try integrating with a native function calling endpoint.
# 
# And then we'll plug this into a RAG pipeline to dynamically make decisions on QA vs. summarization.

# ## 1. Setup a Basic Router Prompt
# 
# At its core, a router is a module that takes in a set of choices. Given a user query, it "selects" a relevant choice.
# 
# For simplicity, we'll start with the choices as a set of strings.

from llama_index import PromptTemplate

choices = [
    "Useful for questions related to apples",
    "Useful for questions related to oranges",
]

def get_choice_str(choices):
    choices_str = "\n\n".join(
        [f"{idx+1}. {c}" for idx, c in enumerate(choices)]
    )
    return choices_str

choices_str = get_choice_str(choices)

router_prompt0 = PromptTemplate(
    "Some choices are given below. It is provided in a numbered list (1 to"
    " {num_choices}), where each item in the list corresponds to a"
    " summary.\n---------------------\n{context_list}\n---------------------\nUsing"
    " only the choices above and not prior knowledge, return the top choices"
    " (no more than {max_outputs}, but only select what is needed) that are"
    " most relevant to the question: '{query_str}'\n"
)

# Let's try this prompt on a set of toy questions and see what the output brings.

from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

def get_formatted_prompt(query_str):
    fmt_prompt = router_prompt0.format(
        num_choices=len(choices),
        max_outputs=2,
        context_list=choices_str,
        query_str=query_str,
    )
    return fmt_prompt

query_str = "Can you tell me more about the amount of Vitamin C in apples"
fmt_prompt = get_formatted_prompt(query_str)
response = llm.complete(fmt_prompt)

print(str(response))

query_str = "What are the health benefits of eating orange peels?"
fmt_prompt = get_formatted_prompt(query_str)
response = llm.complete(fmt_prompt)

print(str(response))

query_str = (
    "Can you tell me more about the amount of Vitamin C in apples and oranges."
)
fmt_prompt = get_formatted_prompt(query_str)
response = llm.complete(fmt_prompt)

print(str(response))

# **Observation**: While the response corresponds to the correct choice, it can be hacky to parse into a structured output (e.g. a single integer). We'd need to do some string parsing on the choices to extract out a single number, and make it robust to failure modes.

# ## 2. A Router Prompt that can generate structured outputs
# 
# Therefore the next step is to try to prompt the model to output a more structured representation (JSON). 
# 
# We define an output parser class (`RouterOutputParser`). This output parser will be responsible for both formatting the prompt and also parsing the result into a structured object (an `Answer`).
# 
# We then apply the `format` and `parse` methods of the output parser around the LLM call using the router prompt to generate a structured output.

# ### 2.a Import Answer Class
# 
# We load in the Answer class from our codebase. It's a very simple dataclass with two fields: `choice` and `reason`

from dataclasses import fields
from pydantic import BaseModel
import json

class Answer(BaseModel):
    choice: int
    reason: str

print(json.dumps(Answer.schema(), indent=2))

# ### 2.b Define Router Output Parser

from llama_index.types import BaseOutputParser

FORMAT_STR = """The output should be formatted as a JSON instance that conforms to 
the JSON schema below. 

Here is the output schema:
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "choice": {
        "type": "integer"
      },
      "reason": {
        "type": "string"
      }
    },
    "required": [
      "choice",
      "reason"
    ],
    "additionalProperties": false
  }
}
"""

# If we want to put `FORMAT_STR` as part of an f-string as part of a prompt template, then we'll need to escape the curly braces so that they don't get treated as template variables.

def _escape_curly_braces(input_string: str) -> str:
    # Replace '{' with '{{' and '}' with '}}' to escape curly braces
    escaped_string = input_string.replace("{", "{{").replace("}", "}}")
    return escaped_string

# We now define a simple parsing function to extract out the JSON string from the LLM response (by searching for square brackets)

def _marshal_output_to_json(output: str) -> str:
    output = output.strip()
    left = output.find("[")
    right = output.find("]")
    output = output[left : right + 1]
    return output

# We put these together in our `RouterOutputParser`

from typing import List

class RouterOutputParser(BaseOutputParser):
    def parse(self, output: str) -> List[Answer]:
        """Parse string."""
        json_output = _marshal_output_to_json(output)
        json_dicts = json.loads(json_output)
        answers = [Answer.from_dict(json_dict) for json_dict in json_dicts]
        return answers

    def format(self, prompt_template: str) -> str:
        return prompt_template + "\n\n" + _escape_curly_braces(FORMAT_STR)

# ### 2.c Give it a Try
# 
# We create a function called `route_query` that will take in the output parser, llm, and prompt template and output a structured answer.

output_parser = RouterOutputParser()

from typing import List

def route_query(
    query_str: str, choices: List[str], output_parser: RouterOutputParser
):
    choices_str

    fmt_base_prompt = router_prompt0.format(
        num_choices=len(choices),
        max_outputs=len(choices),
        context_list=choices_str,
        query_str=query_str,
    )
    fmt_json_prompt = output_parser.format(fmt_base_prompt)

    raw_output = llm.complete(fmt_json_prompt)
    parsed = output_parser.parse(str(raw_output))

    return parsed

# ## 3. Perform Routing with a Function Calling Endpoint
# 

# 
# This process can feel a bit messy. Function calling endpoints (e.g. OpenAI) abstract away this complexity by allowing the model to natively output structured functions. This obviates the need to manually prompt + parse the outputs. 
# 
# LlamaIndex offers an abstraction called a `PydanticProgram` that integrates with a function endpoint to produce a structured Pydantic object. We integrate with OpenAI and Guidance.

# We redefine our `Answer` class with annotations, as well as an `Answers` class containing a list of answers.

from pydantic import Field

class Answer(BaseModel):
    "Represents a single choice with a reason."
    choice: int
    reason: str

class Answers(BaseModel):
    """Represents a list of answers."""

    answers: List[Answer]

Answers.schema()

from llama_index.program import OpenAIPydanticProgram

router_prompt1 = router_prompt0.partial_format(
    num_choices=len(choices),
    max_outputs=len(choices),
)

program = OpenAIPydanticProgram.from_defaults(
    output_cls=Answers,
    prompt=router_prompt1,
    verbose=True,
)

query_str = "What are the health benefits of eating orange peels?"
output = program(context_list=choices_str, query_str=query_str)

output

# ## 4. Plug Router Module as part of a RAG pipeline
# 

# ### Setup: Load Data
# 
# We load the Llama 2 paper as data.

#('mkdir data')
#('wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"')

from pathlib import Path
from llama_hub.file.pymu_pdf.base import PyMuPDFReader

loader = PyMuPDFReader()
documents = loader.load(file_path="./data/llama2.pdf")

# ### Setup: Define Indexes
# 
# Define both a vector index and summary index over this data.

from llama_index import ServiceContext, VectorStoreIndex, SummaryIndex

service_context = ServiceContext.from_defaults(chunk_size=1024)
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
summary_index = SummaryIndex.from_documents(
    documents, service_context=service_context
)

vector_query_engine = vector_index.as_query_engine()
summary_query_engine = summary_index.as_query_engine()

# ### Define RouterQueryEngine
# 
# We subclass our `CustomQueryEngine` to define a custom router.

from llama_index.query_engine import CustomQueryEngine, BaseQueryEngine
from llama_index.response_synthesizers import TreeSummarize

class RouterQueryEngine(CustomQueryEngine):
    """Use our Pydantic program to perform routing."""

    query_engines: List[BaseQueryEngine]
    choice_descriptions: List[str]
    verbose: bool = False
    router_prompt: PromptTemplate
    llm: OpenAI
    summarizer: TreeSummarize = Field(default_factory=TreeSummarize)

    def custom_query(self, query_str: str):
        """Define custom query."""

        program = OpenAIPydanticProgram.from_defaults(
            output_cls=Answers,
            prompt=router_prompt1,
            verbose=self.verbose,
            llm=self.llm,
        )

        choices_str = get_choice_str(self.choice_descriptions)
        output = program(context_list=choices_str, query_str=query_str)
        # print choice and reason, and query the underlying engine
        if self.verbose:
            print(f"Selected choice(s):")
            for answer in output.answers:
                print(f"Choice: {answer.choice}, Reason: {answer.reason}")

        responses = []
        for answer in output.answers:
            choice_idx = answer.choice - 1
            query_engine = self.query_engines[choice_idx]
            response = query_engine.query(query_str)
            responses.append(response)

        # if a single choice is picked, we can just return that response
        if len(responses) == 1:
            return responses[0]
        else:
            # if multiple choices are picked, we can pick a summarizer
            response_strs = [str(r) for r in responses]
            result_response = self.summarizer.get_response(
                query_str, response_strs
            )
            return result_response

choices = [
    (
        "Useful for answering questions about specific sections of the Llama 2"
        " paper"
    ),
    "Useful for questions that ask for a summary of the whole paper",
]

router_query_engine = RouterQueryEngine(
    query_engines=[vector_query_engine, summary_query_engine],
    choice_descriptions=choices,
    verbose=True,
    router_prompt=router_prompt1,
    llm=OpenAI(model="gpt-4"),
)

# ### Try our constructed Router Query Engine
# 
# Let's take our self-built router query engine for a spin! We ask a question that routes to the vector query engine, and also another question that routes to the summarization engine.

response = router_query_engine.query(
    "How does the Llama 2 model compare to GPT-4 in the experimental results?"
)

print(str(response))

response = router_query_engine.query("Can you give a summary of this paper?")

print(str(response))

