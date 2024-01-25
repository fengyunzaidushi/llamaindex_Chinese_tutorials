#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/query_engine/pandas_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Pandas Query Engine

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys
from IPython.#display import Markdown, #display

import pandas as pd
from llama_index.query_engine import PandasQueryEngine

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# ### Let's start on a Toy DataFrame
# 
# Very simple dataframe containing city and population pairs.

# Test on some sample data
df = pd.DataFrame(
    {
        "city": ["Toronto", "Tokyo", "Berlin"],
        "population": [2930000, 13960000, 3645000],
    }
)

query_engine = PandasQueryEngine(df=df, verbose=True)

response = query_engine.query(
    "What is the city with the highest population?",
)

#display(Markdown(f"<b>{response}</b>"))

# get pandas python instructions
print(response.metadata["pandas_instruction_str"])

# ### Analyzing the Titanic Dataset
# 
# The Titanic dataset is one of the most popular tabular datasets in introductory machine learning
# Source: https://www.kaggle.com/c/titanic

# #### Download Data

#("wget 'https://raw.githubusercontent.com/jerryjliu/llama_index/main/docs/examples/data/csv/titanic_train.csv' -O 'titanic_train.csv'")

df = pd.read_csv("./titanic_train.csv")

query_engine = PandasQueryEngine(df=df, verbose=True)

response = query_engine.query(
    "What is the correlation between survival and age?",
)

#display(Markdown(f"<b>{response}</b>"))

# get pandas python instructions
print(response.metadata["pandas_instruction_str"])

