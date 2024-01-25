#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/response_synthesizers/structured_refine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Refine with Structured Answer Filtering
# When using our Refine response synthesizer for response synthesis, it's crucial to filter out non-answers. An issue often encountered is the propagation of a single unhelpful response like "I don't have the answer", which can persist throughout the synthesis process and lead to a final answer of the same nature. This can occur even when there are actual answers present in other, more relevant sections.
# 
# These unhelpful responses can be filtered out by setting `structured_answer_filtering` to `True`. It is set to `False` by default since this currently only works best if you are using an OpenAI model that supports function calling.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## Load Data

texts = [
    "The president in the year 2040 is John Cena.",
    "The president in the year 2050 is Florence Pugh.",
    'The president in the year 2060 is Dwayne "The Rock" Johnson.',
]

# ## Summarize

from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-0613")

from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(llm=llm)

from llama_index.response_synthesizers import get_response_synthesizer

summarizer = get_response_synthesizer(
    response_mode="refine", service_context=service_context, verbose=True
)

response = summarizer.get_response("who is president in the year 2050?", texts)

# ### Failed Result
# As you can see, we weren't able to get the correct answer from the input `texts` strings since the initial "I don't know" answer propogated through till the end of the response synthesis.

print(response)

# Now we'll try again with `structured_answer_filtering=True`

from llama_index.response_synthesizers import get_response_synthesizer

summarizer = get_response_synthesizer(
    response_mode="refine",
    service_context=service_context,
    verbose=True,
    structured_answer_filtering=True,
)

response = summarizer.get_response("who is president in the year 2050?", texts)

# ### Successful Result
# As you can see, we were able to determine the correct answer from the given context by filtering the `texts` strings for the ones that actually contained the answer to our question.

print(response)

# ## Non Function-calling LLMs
# You may want to make use of this filtering functionality with an LLM that doesn't offer a function calling API.
# 

# we'll stick with OpenAI but use an older model that does not support function calling
davinci_llm = OpenAI(model="text-davinci-003")

from llama_index import ServiceContext
from llama_index.response_synthesizers import get_response_synthesizer

davinci_service_context = ServiceContext.from_defaults(llm=davinci_llm)

summarizer = get_response_synthesizer(
    response_mode="refine",
    service_context=davinci_service_context,
    verbose=True,
    structured_answer_filtering=True,
)

response = summarizer.get_response("who is president in the year 2050?", texts)
print(response)

# ### `CompactAndRefine`
# Since `CompactAndRefine` is built on top of `Refine`, this response mode also supports structured answer filtering.

from llama_index.response_synthesizers import get_response_synthesizer

summarizer = get_response_synthesizer(
    response_mode="compact",
    service_context=service_context,
    verbose=True,
    structured_answer_filtering=True,
)

response = summarizer.get_response("who is president in the year 2050?", texts)
print(response)

