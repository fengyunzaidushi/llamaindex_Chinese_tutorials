#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/output_parsing/lmformatenforcer_regular_expressions.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # LM Format Enforcer Regular Expression Generation

# Generate structured data with [**lm-format-enforcer**](https://github.com/noamgat/lm-format-enforcer) via LlamaIndex.  
# 
# 
# With lm-format-enforcer, you can guarantee the output structure is correct by *forcing* the LLM to output desired tokens.  
# This is especialy helpful when you are using lower-capacity model (e.g. the current open source models), which otherwise would struggle to generate valid output that fits the desired output schema.
# 
# [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) supports regular expressions and JSON Schema, this demo focuses on regular expressions. For JSON Schema + Pydantic, see the [sample Pydantic program notebook](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/output_parsing/lmformatenforcer_pydantic_program.ipynb).

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index lm-format-enforcer llama-cpp-python')

import lmformatenforcer
import re

from llama_index.prompts.lmformatenforcer_utils import (
    activate_lm_format_enforcer,
    build_lm_format_enforcer_function,
)

# Define output format

regex = r'"Hello, my name is (?P<name>[a-zA-Z]*)\. I was born in (?P<hometown>[a-zA-Z]*). Nice to meet you!"'

# Create the model. We use `LlamaCPP` as the LLM in this demo, but `HuggingFaceLLM` is also supported.

from llama_index.llms.llama_cpp import LlamaCPP

llm = LlamaCPP()

# Activate the format enforcer and run the LLM get structured output in the desired regular expression format. As long as we are inside the `with activate_lm_format_enforcer(...)` block, the LLM will output the desired format.
# 
# If we would have used `lmformatenforcer.JsonSchemaParser` and a JSON schema, we would have gotten JSON output instead.

regex_parser = lmformatenforcer.RegexParser(regex)
lm_format_enforcer_fn = build_lm_format_enforcer_function(llm, regex_parser)
with activate_lm_format_enforcer(llm, lm_format_enforcer_fn):
    output = llm.complete(
        "Here is a way to present myself, if my name was John and I born in Boston: "
    )

# The output is a string, according to the regular expression, which we can parse and extract parameters from.

print(output)
print(re.match(regex, output.text).groupdict())

