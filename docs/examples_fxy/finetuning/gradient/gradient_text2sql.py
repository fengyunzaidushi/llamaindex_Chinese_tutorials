#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/finetuning/gradient/gradient_text2sql.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Fine Tuning for Text-to-SQL With Gradient and LlamaIndex
# 

# 
# We do this by using [gradient.ai](https://gradient.ai)
# 
# **NOTE**: This is an alternative to our repo/guide on fine-tuning llama2-7b with Modal: https://github.com/run-llama/modal_finetune_sql

#('pip install llama-index gradientai -q')

import os
from llama_index.llms import GradientBaseModelLLM
from llama_index.finetuning.gradient.base import GradientFinetuneEngine

os.environ["GRADIENT_ACCESS_TOKEN"] = os.getenv("GRADIENT_API_KEY")
os.environ["GRADIENT_WORKSPACE_ID"] = ""

# ## Prepare Data
# 
# We load sql-create-context from Hugging Face datasets. The dataset is a mix of WikiSQL and Spider, and is organized in the format of input query, context, and ground-truth SQL statement. The context is a CREATE TABLE statement.

dialect = "sqlite"

# #### Load Data, Save to a Directory

from datasets import load_dataset
from pathlib import Path
import json

def load_jsonl(data_dir):
    data_path = Path(data_dir).as_posix()
    data = load_dataset("json", data_files=data_path)
    return data

def save_jsonl(data_dicts, out_path):
    with open(out_path, "w") as fp:
        for data_dict in data_dicts:
            fp.write(json.dumps(data_dict) + "\n")

def load_data_sql(data_dir: str = "data_sql"):
    dataset = load_dataset("b-mc2/sql-create-context")

    dataset_splits = {"train": dataset["train"]}
    out_path = Path(data_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for key, ds in dataset_splits.items():
        with open(out_path, "w") as f:
            for item in ds:
                newitem = {
                    "input": item["question"],
                    "context": item["context"],
                    "output": item["answer"],
                }
                f.write(json.dumps(newitem) + "\n")

# dump data to data_sql
load_data_sql(data_dir="data_sql")

# #### Split into Training/Validation Splits

from math import ceil

def get_train_val_splits(
    data_dir: str = "data_sql",
    val_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
):
    data = load_jsonl(data_dir)
    num_samples = len(data["train"])
    val_set_size = ceil(val_ratio * num_samples)

    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=shuffle, seed=seed
    )
    return train_val["train"].shuffle(), train_val["test"].shuffle()

raw_train_data, raw_val_data = get_train_val_splits(data_dir="data_sql")
save_jsonl(raw_train_data, "train_data_raw.jsonl")
save_jsonl(raw_val_data, "val_data_raw.jsonl")

raw_train_data[0]

# #### Map Training/Dataset Dictionaries to Prompts
# 
# Here we define functions to map the dataset dictionaries to a prompt format, that we can then feed to gradient.ai's fine-tuning endpoint.

### Format is similar to the nous-hermes LLMs

text_to_sql_tmpl_str = """\
<s>##

text_to_sql_inference_tmpl_str = """\
<s>##

### Alternative Format
### Recommended by gradient.ai docs, but empirically we found worse results here

# text_to_sql_tmpl_str = """\
# <s>[INST] SYS\n{system_message}\n<</SYS>>\n\n{user_message} [/INST] {response} </s>"""

# text_to_sql_inference_tmpl_str = """\
# <s>[INST] SYS\n{system_message}\n<</SYS>>\n\n{user_message} [/INST] """

def _generate_prompt_sql(input, context, dialect="sqlite", output=""):
    system_message = f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. 

You must output the SQL query that answers the question.
    
    """
    user_message = f"""### Dialect:
{dialect}

##
{input}

### Context:
{context}

### Response:
"""
    if output:
        return text_to_sql_tmpl_str.format(
            system_message=system_message,
            user_message=user_message,
            response=output,
        )
    else:
        return text_to_sql_inference_tmpl_str.format(
            system_message=system_message, user_message=user_message
        )

def generate_prompt(data_point):
    full_prompt = _generate_prompt_sql(
        data_point["input"],
        data_point["context"],
        dialect="sqlite",
        output=data_point["output"],
    )
    return {"inputs": full_prompt}

train_data = [
    {"inputs": d["inputs"] for d in raw_train_data.map(generate_prompt)}
]
save_jsonl(train_data, "train_data.jsonl")
val_data = [{"inputs": d["inputs"] for d in raw_val_data.map(generate_prompt)}]
save_jsonl(val_data, "val_data.jsonl")

print(train_data[0]["inputs"])

# ## Run Fine-tuning with gradient.ai
# 
# Here we call Gradient's fine-tuning endpoint with the `GradientFinetuneEngine`. 
# 
# We limit the steps for example purposes, but feel free to modify the parameters as you wish. 
# 
# At the end we fetch our fine-tuned LLM.

# base_model_slug = "nous-hermes2"
base_model_slug = "llama2-7b-chat"
base_llm = GradientBaseModelLLM(
    base_model_slug=base_model_slug, max_tokens=300
)

# step max steps to 20 just for testing purposes
# NOTE: can only specify one of base_model_slug or model_adapter_id
finetune_engine = GradientFinetuneEngine(
    base_model_slug=base_model_slug,
    # model_adapter_id='805c6fd6-daa8-4fc8-a509-bebb2f2c1024_model_adapter',
    name="text_to_sql",
    data_path="train_data.jsonl",
    verbose=True,
    max_steps=200,
    batch_size=4,
)

finetune_engine.model_adapter_id

epochs = 1
for i in range(epochs):
    print(f"** EPOCH {i} **")
    finetune_engine.finetune()

ft_llm = finetune_engine.get_finetuned_model(max_tokens=300)

# ## Evaluation
# 
# This is two parts:
# 1. We evaluate on some sample datapoints in the validation dataset.
# 2. We evaluate on a new toy SQL dataset, and plug the fine-tuned LLM into our `NLSQLTableQueryEngine` to run a full text-to-SQL workflow.
# 
# 

# ### Part 1: Evaluation on Validation Dataset Datapoints

from llama_index import ServiceContext

def get_text2sql_completion(llm, raw_datapoint):
    service_context = ServiceContext.from_defaults(llm=llm)
    text2sql_tmpl_str = _generate_prompt_sql(
        raw_datapoint["input"],
        raw_datapoint["context"],
        dialect="sqlite",
        output=None,
    )

    response = llm.complete(text2sql_tmpl_str)
    return str(response)

test_datapoint = raw_val_data[2]
#display(test_datapoint)

# run base llama2-7b-chat model
get_text2sql_completion(base_llm, test_datapoint)

# run fine-tuned llama2-7b-chat model
get_text2sql_completion(ft_llm, test_datapoint)

# ### Part 2: Evaluation on a Toy Dataset
# 
# Here we create a toy table of cities and their populations.

# #### Create Table

# create sample
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)
from llama_index import SQLDatabase

engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

# create city SQL table
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)
metadata_obj.create_all(engine)

# This context is used later on
from sqlalchemy.schema import CreateTable

table_create_stmt = str(CreateTable(city_stats_table))
print(table_create_stmt)

sql_database = SQLDatabase(engine, include_tables=["city_stats"])

# #### Populate with Test Datapoints

# insert sample rows
from sqlalchemy import insert

rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {
        "city_name": "Chicago",
        "population": 2679000,
        "country": "United States",
    },
    {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.connect() as connection:
        cursor = connection.execute(stmt)
        connection.commit()

# #### Get Text2SQL Query Engine

from llama_index.query_engine import NLSQLTableQueryEngine
from llama_index import ServiceContext, PromptTemplate

def get_text2sql_query_engine(llm, table_context, sql_database):
    service_context = ServiceContext.from_defaults(llm=llm)
    # we essentially swap existing template variables for new template variables
    # to put into our `NLSQLTableQueryEngine`
    text2sql_tmpl_str = _generate_prompt_sql(
        "{query_str}", "{schema}", dialect="{dialect}", output=""
    )
    sql_prompt = PromptTemplate(text2sql_tmpl_str)
    # Here we explicitly set the table context to be the CREATE TABLE string
    # So we set `tables` to empty, and hard fix `context_str` prefix

    query_engine = NLSQLTableQueryEngine(
        sql_database,
        tables=[],
        context_str_prefix=table_context,
        text_to_sql_prompt=sql_prompt,
        service_context=service_context,
        synthesize_response=False,
    )
    return query_engine

# query = "Which cities have populations less than 10 million people?"
query = "What is the population of Tokyo? (make sure cities/countries are capitalized)"
# query = "What is the average population and total population of the cities?"

# #### Results with base llama2 model
# The base llama2 model appends a bunch of text to the SQL statement that breaks our parser (and has minor capitalization mistakes)

base_query_engine = get_text2sql_query_engine(
    base_llm, table_create_stmt, sql_database
)

base_response = base_query_engine.query(query)

print(str(base_response))

base_response.metadata["sql_query"]

# #### Results with fine-tuned model

ft_query_engine = get_text2sql_query_engine(
    ft_llm, table_create_stmt, sql_database
)

ft_response = ft_query_engine.query(query)

print(str(ft_response))

ft_response.metadata["sql_query"]

