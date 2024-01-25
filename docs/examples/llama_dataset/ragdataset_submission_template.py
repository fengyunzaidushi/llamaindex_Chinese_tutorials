#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/llama_datasets/ragdataset_submission_template.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# <a id='top'></a>
# # `LlamaDataset` Submission Template Notebook
# 
# This notebook serves as a template for creating a particular kind of `LlamaDataset`, namely `LabelledRagDataset`. Additionally, this template aids in the preparation of all of the necessary supplementary materials in order to make a `LlamaDataset` contribution to [llama-hub](https://llamahub.ai).
# 
# **NOTE**: Since this notebook uses OpenAI LLM's as a default, an `OPENAI_API_KEY` is required. You can pass the `OPENAI_API_KEY` by specifying the `api_key` argument when constructing the LLM. Or by running `export OPENAI_API_KEY=<api_key>` before spinning up this jupyter notebook.

# ### Prerequisites

# #### Fork and Clone Required Github Repositories
# 
# Contributing a `LlamaDataset` to `llama-hub` is similar to contributing any of the other `llama-hub` artifacts (`LlamaPack`, `Tool`, `Loader`), in that you'll be required to make a contribution to the [llama-hub repository](https://github.com/run-llama/llama-hub). However, unlike for those other artifacts, for a `LlamaDataset`, you'll also be required to make a contribution to another Github repository, namely the [llama-datasets repository](https://github.com/run-llama/llama-datasets).
# 
# 1. Fork and clone `llama-hub` Github repository
# ```bash
# git clone git@github.com:<your-github-user-name>/llama-hub.git  # for ssh
# git clone https://github.com/<your-github-user-name>/llama-hub.git  # for https
# ```
# 2. Fork and clone `llama-datasets` Github repository. **NOTE**: this is a Github LFS repository, and so, when cloning the repository **please ensure that you prefix the clone command with** `GIT_LFS_SKIP_SMUDGE=1` in order to not download any of the large data files.
# ```bash
# # for bash
# GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:<your-github-user-name>/llama-datasets.git  # for ssh
# GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/<your-github-user-name>/llama-datasets.git  # for https
# 
# # for windows its done in two commands
# set GIT_LFS_SKIP_SMUDGE=1  
# git clone git@github.com:<your-github-user-name>/llama-datasets.git  # for ssh
# 
# set GIT_LFS_SKIP_SMUDGE=1  
# git clone https://github.com/<your-github-user-name>/llama-datasets.git  # for https
# ```

# #### A Quick Primer on `LabelledRagDataset` and `LabelledRagDataExample`
# 
# A `LabelledRagDataExample` is a Pydantic `BaseModel` that contains the following fields:
# - `query` representing the question or query of the example
# - `query_by` notating whether the query was human generated or ai generated
# - `reference_answer` representing the reference (ground-truth) answer to the query
# - `reference_answer_by` notating whether the reference answer was human generated or ai generated
# - `reference_contexts` an optional list of text strings representing the contexts used in generating the reference answer
# 
# A `LabelledRagDataset` is also a Pydantic `BaseModel` that contains the lone field:
# - `examples` is a list of `LabelledRagDataExample`'s
# 

# ## Steps For Making A `LlamaDataset` Submission
# 
# (NOTE: these links are only functional while in the notebook.)
# 
# 1. Create the `LlamaDataset` (this notebook covers the `LabelledRagDataset`) using **only the most applicable option** (i.e., one) of the three listed below:
#     1. [From scratch and synthetically constructed examples](#1A)
#     2. [From an existing and similarly structured question-answer dataset](#1B)
#     3. [From scratch and manually constructed examples](#1C)
# 2. [Generate a baseline evaluation result](#Step2)
# 3. [Prepare `card.json` and `README.md`](#Step3) by doing **only one** of either of the listed options below:
#     1. [Automatic generation with `LlamaDatasetMetadataPack`](#3A)
#     2. [Manual generation](#3B)
# 5. [Submit a pull-request into the `llama-hub` repository to register the `LlamaDataset`](#Step4)
# 7. [Submit a pull-request into the `llama-datasets` repository to upload the `LlamaDataset` and its source files](#Step5)

# <a id='1A'></a>
# ## 1A. Creating a `LabelledRagDataset` from scratch with synthetically constructed examples
# 
# Use the code template below to construct your examples from scratch and synthetic data generation. In particular, we load a source text as a set of `Document`'s, and then use an LLM to generate question and answer pairs to construct our dataset.

# #### Demonstration

# NESTED ASYNCIO LOOP NEEDED TO RUN ASYNC IN A NOTEBOOK
import nest_asyncio

nest_asyncio.apply()

# DOWNLOAD RAW SOURCE DATA
#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

from llama_index.readers import SimpleDirectoryReader
from llama_index.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms import OpenAI
from llama_index import ServiceContext

# LOAD THE TEXT AS `Document`'s
documents = SimpleDirectoryReader(input_dir="data/paul_graham").load_data()

# USE `RagDatasetGenerator` TO PRODUCE A `LabelledRagDataset`
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(llm=llm)

dataset_generator = RagDatasetGenerator.from_documents(
    documents,
    service_context=service_context,
    num_questions_per_chunk=2,  # set the number of questions per nodes
    show_progress=True,
)

rag_dataset = dataset_generator.generate_dataset_from_nodes()

rag_dataset.to_pandas()[:5]

# #### Template

from llama_index.readers import SimpleDirectoryReader
from llama_index.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms import OpenAI
from llama_index import ServiceContext

documents = SimpleDirectoryReader(input_dir=<FILL-IN>).load_data()
llm=<FILL-IN>  # Recommend OpenAI GPT-4 for reference_answer generation
service_context = ServiceContext.from_defaults(llm=llm)

dataset_generator = RagDatasetGenerator.from_documents(
    documents,
    service_context=service_context,
    num_questions_per_chunk=<FILL-IN>,  # set the number of questions per nodes
    show_progress=True,
)

rag_dataset = dataset_generator.generate_dataset_from_nodes()

# save this dataset as it is required for the submission
rag_dataset.save_json("rag_dataset.json")

# #### [Step 2](#Step2), [Back to top](#top) 

# <a id='1B'></a>
# ## 1B. Creating a `LabelledRagDataset` from an existing and similarly structured dataset
# 
# Follow the demonstration and use the provided template to convert a question-answer dataset loaded as a pandas `DataFrame` into a `LabelledRagDataset`. As a demonstration, we will load in the generative part of the [TruthfulQA dataset](https://huggingface.co/datasets/truthful_qa).

# #### Demonstration

#("mkdir -p 'data/truthfulqa/'")
#('wget "https://raw.githubusercontent.com/sylinrl/TruthfulQA/013686a06be7a7bde5bf8223943e106c7250123c/TruthfulQA.csv" -O "data/truthfulqa/truthfulqa.csv"')

import pandas as pd

source_df = pd.read_csv("data/truthfulqa/truthfulqa.csv")
source_df.head()

# ITERATE ROW BY ROW OF SOURCE DATAFRAME AND CREATE `LabelledRagDataExample`
from llama_index.llama_dataset import (
    LabelledRagDataExample,
    CreatedBy,
    CreatedByType,
)
from llama_index.llama_dataset import LabelledRagDataset

examples = []
for ix, row in source_df.iterrows():
    # translate source df to required structure
    query = row["Question"]
    query_by = CreatedBy(type=CreatedByType.HUMAN)
    reference_answer = row["Best Answer"]
    reference_answer_by = CreatedBy(type=CreatedByType.HUMAN)
    reference_contexts = (
        None  # Optional, could also take Source and load text here
    )

    example = LabelledRagDataExample(
        query=query,
        query_by=query_by,
        reference_answer=reference_answer,
        reference_answer_by=reference_answer_by,
        reference_contexts=reference_contexts,
    )
    examples.append(example)

rag_dataset = LabelledRagDataset(examples=examples)

rag_dataset.to_pandas()[:5]

# #### Template

import pandas as pd
from llama_index.llama_dataset import (
    LabelledRagDataExample,
    CreatedBy,
    CreatedByType,
)
from llama_index.llama_dataset import LabelledRagDataset

source_df = <FILL-IN>

examples = []
for ix, row in source_df.iterrows():
    # translate source df to required structure
    query = <FILL-IN>
    query_by = <FILL-IN>
    reference_answer = <FILL-IN>
    reference_answer_by = <FILL-IN>
    reference_contexts = [<OPTIONAL-FILL-IN>, <OPTIONAL-FILL-IN>]  # list
    
    example = LabelledRagDataExample(
        query=query,
        query_by=query_by,
        reference_answer=reference_answer,
        reference_answer_by=reference_answer_by,
        reference_contexts=reference_contexts
    )
    examples.append(example)

rag_dataset = LabelledRagDataset(examples=examples)

# save this dataset as it is required for the submission
rag_dataset.save_json("rag_dataset.json")

# #### [Step 2](#Step2), [Back to top](#top) 

# <a id='1C'></a>
# ## 1C. Creating a `LabelledRagDataset` from scratch with manually constructed examples
# 
# Use the code template below to construct your examples from scratch. This method for creating a `LablledRagDataset` is the least scalable out of all the methods shown here. Nonetheless, we include it in this guide for completeness sake, but rather recommend that you use one of two the previous methods instead. Similar to the demonstration for [1A](#1A), we consider the Paul Graham Essay dataset here as well.

# #### Demonstration: 

# DOWNLOAD RAW SOURCE DATA
#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# LOAD TEXT FILE
with open("data/paul_graham/paul_graham_essay.txt", "r") as f:
    raw_text = f.read(700)  # loading only the first 700 characters

print(raw_text)

# MANUAL CONSTRUCTION OF EXAMPLES
from llama_index.llama_dataset import (
    LabelledRagDataExample,
    CreatedBy,
    CreatedByType,
)
from llama_index.llama_dataset import LabelledRagDataset

example1 = LabelledRagDataExample(
    query="Why were Paul's stories awful?",
    query_by=CreatedBy(type=CreatedByType.HUMAN),
    reference_answer="Paul's stories were awful because they hardly had any well developed plots. Instead they just had characters with strong feelings.",
    reference_answer_by=CreatedBy(type=CreatedByType.HUMAN),
    reference_contexts=[
        "I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep."
    ],
)

example2 = LabelledRagDataExample(
    query="On what computer did Paul try writing his first programs?",
    query_by=CreatedBy(type=CreatedByType.HUMAN),
    reference_answer="The IBM 1401.",
    reference_answer_by=CreatedBy(type=CreatedByType.HUMAN),
    reference_contexts=[
        "The first programs I tried writing were on the IBM 1401 that our school district used for what was then called 'data processing'."
    ],
)

# CREATING THE DATASET FROM THE EXAMPLES
rag_dataset = LabelledRagDataset(examples=[example1, example2])

rag_dataset.to_pandas()

rag_dataset[0]  # slicing and indexing supported on `examples` attribute

# #### Template

# MANUAL CONSTRUCTION OF EXAMPLES
from llama_index.llama_dataset import (
    LabelledRagDataExample,
    CreatedBy,
    CreatedByType,
)
from llama_index.llama_dataset import LabelledRagDataset

example1 = LabelledRagDataExample(
    query=<FILL-IN>,
    query_by=CreatedBy(type=CreatedByType.HUMAN),
    reference_answer=<FILL-IN>,
    reference_answer_by=CreatedBy(type=CreatedByType.HUMAN),
    reference_contexts=[<OPTIONAL-FILL-IN>, <OPTIONAL-FILL-IN>],
)

example2 = LabelledRagDataExample(
    query=#<FILL-IN>,
    query_by=CreatedBy(type=CreatedByType.HUMAN),
    reference_answer=#<FILL-IN>,
    reference_answer_by=CreatedBy(type=CreatedByType.HUMAN),
    reference_contexts=#[<OPTIONAL-FILL-IN>],
)

# ... and so on

rag_dataset = LabelledRagDataset(examples=[example1, example2,])

# save this dataset as it is required for the submission
rag_dataset.save_json("rag_dataset.json")

# #### [Back to top](#top) 

# <a id='Step2'></a>
# ## 2. Generate A Baseline Evaluation Result
# 
# Submitting a dataset also requires submitting a baseline result. At a high-level, generating a baseline result comprises of the following steps:
# 
#     i. Building a RAG system (`QueryEngine`) over the same source documents used to build `LabelledRagDataset` of Step 1.
#     ii. Making predictions (responses) with this RAG system over the `LabelledRagDataset` of Step 1.
#     iii. Evaluating the predictions
# 
# It is recommended to carry out Steps ii. and iii. via the `RagEvaluatorPack` which can be downloaded from `llama-hub`.
# 
# **NOTE**: The `RagEvaluatorPack` uses GPT-4 by default as it is an LLM that has demonstrated high alignment with human evaluations.

# #### Demonstration
# This is a demo for 1A, but it would follow similar steps for 1B and 1C.

from llama_index.readers import SimpleDirectoryReader
from llama_index import VectorStoreIndex
from llama_index.llama_pack import download_llama_pack

# i. Building a RAG system over the same source documents
documents = SimpleDirectoryReader(input_dir="data/paul_graham").load_data()
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()

# ii. and iii. Predict and Evaluate using `RagEvaluatorPack`
RagEvaluatorPack = download_llama_pack("RagEvaluatorPack", "./pack")
rag_evaluator = RagEvaluatorPack(
    query_engine=query_engine,
    rag_dataset=rag_dataset,  # defined in 1A
    show_progress=True,
)

############################################################################
# NOTE: If have a lower tier subscription for OpenAI API like Usage Tier 1 #
# then you'll need to use different batch_size and sleep_time_in_seconds.  #
# For Usage Tier 1, settings that seemed to work well were batch_size=5,   #
# and sleep_time_in_seconds=15 (as of December 2023.)                      #
############################################################################

benchmark_df = await rag_evaluator_pack.arun(
    batch_size=20,  # batches the number of openai api calls to make
    sleep_time_in_seconds=1,  # seconds to sleep before making an api call
)

benchmark_df

# #### Template

from llama_index.readers import SimpleDirectoryReader
from llama_index import VectorStoreIndex
from llama_index.llama_pack import download_llama_pack

documents = SimpleDirectoryReader(  # Can use a different reader here.
    input_dir=<FILL-IN>  # Should read the same source files used to create
).load_data()            # the LabelledRagDataset of Step 1.
                       
index = VectorStoreIndex.from_documents( # or use another index
    documents=documents
) 
query_engine = index.as_query_engine()

RagEvaluatorPack = download_llama_pack(
  "RagEvaluatorPack", "./pack"
)
rag_evaluator = RagEvaluatorPack(
    query_engine=query_engine,
    rag_dataset=rag_dataset,  # defined in Step 1A
    judge_llm=<FILL-IN>  # if you rather not use GPT-4
)
benchmark_df = await rag_evaluator.arun()
benchmark_df

# #### [Back to top](#top) 

# <a id='Step3'></a>
# ## 3. Prepare `card.json` and `README.md`
# 
# Submitting a dataset includes the submission of some metadata as well. This metadata lives in two different files, `card.json` and `README.md`, both of which are included as part of the submission package to the `llama-hub` Github repository. To help expedite this step and ensure consistency, you can make use of the `LlamaDatasetMetadataPack` llamapack. Alternatively, you can do this step manually following the demonstration and using the templates provided below.

# <a id='3A'></a>
# ### 3A. Automatic generation with `LlamaDatasetMetadataPack`

# #### Demonstration
# 
# This continues the Paul Graham Essay demonstration example of 1A.

from llama_index.llama_pack import download_llama_pack

LlamaDatasetMetadataPack = download_llama_pack(
    "LlamaDatasetMetadataPack", "./pack"
)

metadata_pack = LlamaDatasetMetadataPack()

dataset_description = (
    "A labelled RAG dataset based off an essay by Paul Graham, consisting of "
    "queries, reference answers, and reference contexts."
)

# this creates and saves a card.json and README.md to the same
# directory where you're running this notebook.
metadata_pack.run(
    name="Paul Graham Essay Dataset",
    description=dataset_description,
    rag_dataset=rag_dataset,
    index=index,
    benchmark_df=benchmark_df,
    baseline_name="llamaindex",
)

# if you want to quickly view these two files, set take_a_peak to True
take_a_peak = False

if take_a_peak:
    import json

    with open("card.json", "r") as f:
        card = json.load(f)

    with open("README.md", "r") as f:
        readme_str = f.read()

    print(card)
    print("\n")
    print(readme_str)

# #### Template

from llama_index.llama_pack import download_llama_pack

LlamaDatasetMetadataPack = download_llama_pack(
  "LlamaDatasetMetadataPack", "./pack"
)

metadata_pack = LlamaDatasetMetadataPack()
metadata_pack.run(
    name=<FILL-IN>,
    description=<FILL-IN>,
    rag_dataset=rag_dataset,  # from step 1
    index=index,  # from step 2
    benchmark_df=benchmark_df,  # from step 2
    baseline_name="llamaindex",  # optionally use another one
    source_urls=<OPTIONAL-FILL-IN>
    code_url=<OPTIONAL-FILL-IN>  # if you wish to submit code to replicate baseline results
)

# After running the above code, you can inspect both `card.json` and `README.md` and make any necessary edits manually before submitting to `llama-hub` Github repository.

# #### [Step 4](#Step4), [Back to top](#top) 

# <a id='3B'></a>
# 
# ### 3B. Manual generation

# 
# #### `card.json`

# #### Demonstration

# ```json
# {
#     "name": "Paul Graham Essay",
#     "description": "A labelled RAG dataset based off an essay by Paul Graham, consisting of queries, reference answers, and reference contexts.",
#     "numberObservations": 44,
#     "containsExamplesByHumans": false,
#     "containsExamplesByAI": true,
#     "sourceUrls": [
#         "http://www.paulgraham.com/articles.html"
#     ],
#     "baselines": [
#         {
#             "name": "llamaindex",
#             "config": {
#                 "chunkSize": 1024,
#                 "llm": "gpt-3.5-turbo",
#                 "similarityTopK": 2,
#                 "embedModel": "text-embedding-ada-002"
#             },
#             "metrics": {
#                 "contextSimilarity": 0.934,
#                 "correctness": 4.239,
#                 "faithfulness": 0.977,
#                 "relevancy": 0.977
#             },
#             "codeUrl": "https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_datasets/paul_graham_essay/llamaindex_baseline.py"
#         }
#     ]
# }
# ```

# #### Template

# ```
# {
#     "name": <FILL-IN>,
#     "description": <FILL-IN>,
#     "numberObservations": <FILL-IN>,
#     "containsExamplesByHumans": <FILL-IN>,
#     "containsExamplesByAI": <FILL-IN>,
#     "sourceUrls": [
#         <FILL-IN>,
#     ],
#     "baselines": [
#         {
#             "name": <FILL-IN>,
#             "config": {
#                 "chunkSize": <FILL-IN>,
#                 "llm": <FILL-IN>,
#                 "similarityTopK": <FILL-IN>,
#                 "embedModel": <FILL-IN>
#             },
#             "metrics": {
#                 "contextSimilarity": <FILL-IN>,
#                 "correctness": <FILL-IN>,
#                 "faithfulness": <FILL-IN>,
#                 "relevancy": <FILL-IN>
#             },
#             "codeUrl": <OPTIONAL-FILL-IN>
#         }
#     ]
# }
# ```

# #### `README.md`
# 

# #### Demonstration
# 
# Click [here](https://raw.githubusercontent.com/run-llama/llama-hub/main/llama_hub/llama_datasets/paul_graham_essay/README.md) for an example `README.md`.

# #### Template

# Click [here](https://raw.githubusercontent.com/run-llama/llama-hub/main/llama_hub/llama_datasets/template_README.md) for a template of `README.md`. Simply copy and paste the contents of that file and replace the placeholders "[NAME]" and "[NAME-CAMELCASE]" with the appropriate values according to your new dataset name choice. For example:
# - "{NAME}" = "Paul Graham Essay Dataset"
# - "{NAME_CAMELCASE}" = PaulGrahamEssayDataset

# #### [Back to top](#top) 

# <a id='Step4'></a>
# ## 4. Submit Pull Request To [llama-hub](https://github.com/run-llama/llama-hub) Repo
# 
# Now, is the time to submit the metadata for your new dataset and make a new entry in the datasets registry, which is stored in the file `library.json` (i.e., see it [here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_datasets/library.json)).
# 
# ### 4a. Create a new directory under `llama_hub/llama_datasets` and add your `card.json` and `README.md`:
# ```bash
# cd llama-hub  # cd into local clone of llama-hub
# cd llama_hub/llama_datasets
# git checkout -b my-new-dataset  # create a new git branch
# mkdir <dataset_name_snake_case>  # follow convention of other datasets
# cd <dataset_name_snake_case>
# vim card.json # use vim or another text editor to add in the contents for card.json
# vim README.md # use vim or another text editor to add in the contents for README.md
# ```

# ### 4b. Create an entry in `llama_hub/llama_datasets/library.json`
# 
# ```bash
# cd llama_hub/llama_datasets
# vim library.json # use vim or another text editor to register your new dataset
# ```

# #### Demonstration of `library.json`
# 
# ```json
#   "PaulGrahamEssayDataset": {
#     "id": "llama_datasets/paul_graham_essay",
#     "author": "nerdai",
#     "keywords": ["rag"]
#   }
# ```

# #### Template of `library.json`
# 
# ```json
#   "<FILL-IN>": {
#     "id": "llama_datasets/<dataset_name_snake_case>",
#     "author": "<FILL-IN>",
#     "keywords": ["rag"]
#   }
# ```
# 
# **NOTE**: Please use the same `dataset_name_snake_case` as used in 4a.

# ### 4c. `git add` and `commit` your changes then push to your fork
# 
# ```bash
# git add .
# git commit -m "my new dataset submission"
# git push origin my-new-dataset
# ```
# 
# After this, head over to the Github page for [llama-hub](https://github.com/run-llama/llama-hub). You should see the option to make a pull request from your fork. Go ahead and do that now.

# #### [Back to top](#top) 

# <a id='Step5'></a>
# ## 5. Submit Pull Request To [llama-datasets](https://github.com/run-llama/llama-datasets) Repo

# ### 5a. Create a new directory under `llama_datasets/`:
# 
# ```bash
# cd llama-datasets # cd into local clone of llama-datasets
# git checkout -b my-new-dataset  # create a new git branch
# mkdir <dataset_name_snake_case>  # use the same name as used in Step 4.
# cd <dataset_name_snake_case>
# cp <path-in-local-machine>/rag_dataset.json .  # add rag_dataset.json
# mkdir source_files  # time to add all of the source files
# cp -r <path-in-local-machine>/source_files  ./source_files  # add all source files
# ```
# 
# **NOTE**: Please use the same `dataset_name_snake_case` as used in Step 4.

# ### 5b. `git add` and `commit` your changes then push to your fork
# 
# ```bash
# git add .
# git commit -m "my new dataset submission"
# git push origin my-new-dataset
# ```
# 
# After this, head over to Github page for [llama-datasets](https://github.com/run-llama/llama-datasets). You should see the option to make a pull request from your fork. Go ahead and do that now.

# #### [Back to top](#top) 

# ## Et Voila !
# 
# You've made it to the end of the dataset submission process! 🎉🦙 Congratulations, and thank you for your contribution!
