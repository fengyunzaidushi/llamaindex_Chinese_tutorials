#!/usr/bin/env python
# coding: utf-8

# # OpenAI JSON Mode vs. Function Calling for Data Extraction 

# OpenAI just released [JSON Mode](https://platform.openai.com/docs/guides/text-generation/json-mode): This new config constrain the LLM to only generate strings that parse into valid JSON (but no guarantee on validation against any schema).
# 
# Before this, the best way to extract structured data from text is via [function calling](https://platform.openai.com/docs/guides/function-calling).  
# 

# 
# *Update*: OpenAI has clarified that JSON mode is always enabled for function calling, it's opt-in for regular messages (https://community.openai.com/t/json-mode-vs-function-calling/476994/4)

# ### Generate synthetic data

# We'll start by generating some synthetic data for our data extraction task. Let's ask our LLM for a hypothetical sales transcript.

from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-1106")
response = llm.complete(
    "Generate a sales call transcript, use real names, talk about a product, discuss some action items"
)

transcript = response.text
print(transcript)

# ### Setup our desired schema

# Let's specify our desired output "shape", as a Pydantic Model.

from pydantic import BaseModel, Field
from typing import List

class CallSummary(BaseModel):
    """Data model for a call summary."""

    summary: str = Field(
        description="High-level summary of the call transcript. Should not exceed 3 sentences."
    )
    products: List[str] = Field(
        description="List of products discussed in the call"
    )
    rep_name: str = Field(description="Name of the sales rep")
    prospect_name: str = Field(description="Name of the prospect")
    action_items: List[str] = Field(description="List of action items")

# ### Data extraction with function calling

# We can use the `OpenAIPydanticProgram` module in LlamaIndex to make things super easy, simply define a prompt template, and pass in the LLM and pydantic model we've definied.

from llama_index.program import OpenAIPydanticProgram
from llama_index.prompts import ChatPromptTemplate, ChatMessage

prompt = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert assitant for summarizing and extracting insights from sales call transcripts."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Here is the transcript: \n"
                "------\n"
                "{transcript}\n"
                "------"
            ),
        ),
    ]
)
program = OpenAIPydanticProgram.from_defaults(
    output_cls=CallSummary,
    llm=llm,
    prompt=prompt,
    verbose=True,
)

output = program(transcript=transcript)

# We now have the desired structured data, as a Pydantic Model. 
# Quick inspection shows that the results are as we expected. 

output.dict()

# ### Data extraction with JSON mode

# Let's try to do the same with JSON mode, instead of function calling

prompt = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert assitant for summarizing and extracting insights from sales call transcripts.\n"
                "Generate a valid JSON following the given schema below:\n"
                "{json_schema}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Here is the transcript: \n"
                "------\n"
                "{transcript}\n"
                "------"
            ),
        ),
    ]
)

messages = prompt.format_messages(
    json_schema=CallSummary.schema_json(), transcript=transcript
)

output = llm.chat(
    messages, response_format={"type": "json_object"}
).message.content

# We get a vaid JSON, but it's only regurgitating the schema we specified, and not actually doing the extraction. 

print(output)

# Let's try again by just showing the JSON format we want, instead of specifying the schema

import json

prompt = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert assitant for summarizing and extracting insights from sales call transcripts.\n"
                "Generate a valid JSON in the following format:\n"
                "{json_example}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Here is the transcript: \n"
                "------\n"
                "{transcript}\n"
                "------"
            ),
        ),
    ]
)

dict_example = {
    "summary": "High-level summary of the call transcript. Should not exceed 3 sentences.",
    "products": ["product 1", "product 2"],
    "rep_name": "Name of the sales rep",
    "prospect_name": "Name of the prospect",
    "action_items": ["action item 1", "action item 2"],
}

json_example = json.dumps(dict_example)

messages = prompt.format_messages(
    json_example=json_example, transcript=transcript
)

output = llm.chat(
    messages, response_format={"type": "json_object"}
).message.content

# Now we are able to get the extracted structured data as we expected.

print(output)

# ### Quick Takeaways

# * Function calling remains easier to use for structured data extraction (especially if you have already specified your schema as e.g. a pydantic model)
# * While JSON mode enforces the format of the output, it does not help with validation against a specified schema. Directly passing in a schema may not generate expected JSON and may require additional careful formatting and prompting.
