#!/usr/bin/env python
# coding: utf-8

# # Fine Tuning GPT-3.5-Turbo
# 

# 
# Specifically, we attempt to distill GPT-4's knowledge, by generating training data with GPT-4 to then fine-tune GPT-3.5.
# 
# All training data is generated using two different sections of our index data, creating both a training and evalution set.
# 
# We then finetune with our `OpenAIFinetuneEngine` wrapper abstraction.
# 
# Evaluation is done using the `ragas` library, which we will detail later on.

# !pip install llama-index pypdf sentence-transformers ragas

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

# ## Data Setup
# 
# Here, we first down load the PDF that we will use to generate training data.

#('curl https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter03.pdf --output IPCC_AR6_WGII_Chapter03.pdf')

# The next step is generating a training and eval dataset.
# 
# We will generate 40 questions on different sections of the PDF we downloaded.
# 
# We can use GPT-3.5 on the eval questions to get our baseline performance.
# 
# Then, we will use GPT-4 on the train questions to generate our training data. The training data will be collected with out `OpenAIFineTuningHandler`.
# 
# This step is entirely optional if you don't want to spend the time/tokens -- the eval and training questions are also provided in this folder, as well as the training data!

# ### Train Generation

from llama_index import SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from llama_index.evaluation import DatasetGenerator

documents = SimpleDirectoryReader(
    input_files=["IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# Shuffle the documents
import random

random.seed(42)
random.shuffle(documents)

gpt_35_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3)
)

question_gen_query = (
    "You are a Teacher/ Professor. Your task is to setup "
    "a quiz/examination. Using the provided context, formulate "
    "a single question that captures an important fact from the "
    "context. Restrict the question to the context information provided."
)

dataset_generator = DatasetGenerator.from_documents(
    documents[:50],
    question_gen_query=question_gen_query,
    service_context=gpt_35_context,
)

# NOTE: this may take some time. Go grab a coffee!
questions = dataset_generator.generate_questions_from_nodes(num=40)
print("Generated ", len(questions), " questions")

with open("train_questions.txt", "w") as f:
    for question in questions:
        f.write(question + "\n")

# ### Eval Generation
# 
# Now, lets generate questions on a completely different set of documents, in order to create our eval dataset.

dataset_generator = DatasetGenerator.from_documents(
    documents[
        50:
    ],  # since we generated ~1 question for 40 documents, we can skip the first 40
    question_gen_query=question_gen_query,
    service_context=gpt_35_context,
)

# NOTE: this may take some time. Go grab a coffee!
questions = dataset_generator.generate_questions_from_nodes(num=40)
print("Generated ", len(questions), " questions")

with open("eval_questions.txt", "w") as f:
    for question in questions:
        f.write(question + "\n")

# #
# 
# For this eval, we will be using the [`ragas` evaluation library](https://github.com/explodinggradients/ragas).
# 
# Ragas has a ton of evaluation metrics for RAG pipelines, and you can read about them [here](https://github.com/explodinggradients/ragas/blob/main/docs/metrics.md).
# 
# For this notebook, we will be using the following two metrics
# 
# - `answer_relevancy` - This measures how relevant is the generated answer to the prompt. If the generated answer is incomplete or contains redundant information the score will be low. This is quantified by working out the chance of an LLM generating the given question using the generated answer. Values range (0,1), higher the better.
# - `faithfulness` - This measures the factual consistency of the generated answer against the given context. This is done using a multi step paradigm that includes creation of statements from the generated answer followed by verifying each of these statements against the context. The answer is scaled to (0,1) range. Higher the better.

questions = []
with open("eval_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())

from llama_index import VectorStoreIndex

# limit the context window to 2048 tokens so that refine is used
gpt_35_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3), context_window=2048
)

index = VectorStoreIndex.from_documents(
    documents, service_context=gpt_35_context
)

query_engine = index.as_query_engine(similarity_top_k=2)

contexts = []
answers = []

for question in questions:
    response = query_engine.query(question)
    contexts.append([x.node.get_content() for x in response.source_nodes])
    answers.append(str(response))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

ds = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
)

result = evaluate(ds, [answer_relevancy, faithfulness])
print(result)

# ## GPT-4 to Collect Training Data
# 
# Here, we use GPT-4 and the `OpenAIFineTuningHandler` to collect data that we want to train on.

from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.callbacks import OpenAIFineTuningHandler
from llama_index.callbacks import CallbackManager

finetuning_handler = OpenAIFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])

gpt_4_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4", temperature=0.3),
    context_window=2048,  # limit the context window artifically to test refine process
    callback_manager=callback_manager,
)

questions = []
with open("train_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())

from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents, service_context=gpt_4_context
)

query_engine = index.as_query_engine(similarity_top_k=2)

for question in questions:
    response = query_engine.query(question)

# ## Create `OpenAIFinetuneEngine`
# 
# We create an `OpenAIFinetuneEngine`: the finetune engine will take care of launching a finetuning job, and returning an LLM model that you can directly plugin to the rest of LlamaIndex workflows.
# 
# We use the default constructor, but we can also directly pass in our finetuning_handler into this engine with the `from_finetuning_handler` class method.
# 
# 

finetuning_handler.save_finetuning_events("finetuning_events.jsonl")

from llama_index.finetuning import OpenAIFinetuneEngine

finetune_engine = OpenAIFinetuneEngine(
    "gpt-3.5-turbo",
    "finetuning_events.jsonl",
    # start_job_id="<start-job-id>"  # if you have an existing job, can specify id here
)

# finetune_engine = OpenAIFinetuneEngine.from_finetuning_handler(
#     finetuning_handler,
#     "gpt-3.5-turbo",
#     "tmp.jsonl"
# )

finetune_engine.finetune()

finetune_engine.get_current_job()

ft_llm = finetune_engine.get_finetuned_model(temperature=0.3)

# ## Evaluation
# 
# After some time, your model will be done training!
# 
# The next step is running our fine-tuned model on our eval dataset again to measure any performance increase.

from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.callbacks import OpenAIFineTuningHandler
from llama_index.callbacks import CallbackManager

# Option 1: pass in ft_llm directly into ServiceContext
ft_context = ServiceContext.from_defaults(
    llm=ft_llm,
    context_window=2048,  # limit the context window artifically to test refine process
)

# # Option 2: you can also specify the model name manually
# ft_model_name = "ft:gpt-3.5-turbo-0613:..."
# ft_context = ServiceContext.from_defaults(
#     llm=OpenAI(model=ft_model_name, temperature=0.3),
#     context_window=2048,  # limit the context window artifically to test refine process
# )

questions = []
with open("eval_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())

from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents, service_context=ft_context)

query_engine = index.as_query_engine(similarity_top_k=2)

contexts = []
answers = []

for question in questions:
    response = query_engine.query(question)
    contexts.append([x.node.get_content() for x in response.source_nodes])
    answers.append(str(response))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

ds = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
)

result = evaluate(ds, [answer_relevancy, faithfulness])
print(result)

# ## Exploring Differences
# 
# Let's quickly compare the differences in responses, to demonstrate that fine tuning did indeed change something.

from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

questions = []
with open("eval_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())

print(questions[12])

# ### Original

from llama_index.response.notebook_utils import #display_response
from llama_index import ServiceContext
from llama_index.llms import OpenAI

gpt_35_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3),
    context_window=2048,  # limit the context window artifically to test refine process
)

query_engine = index.as_query_engine(service_context=gpt_35_context)

response = query_engine.query(questions[12])

#display_response(response)

# ### Fine-Tuned

from llama_index import ServiceContext
from llama_index.llms import OpenAI

ft_context = ServiceContext.from_defaults(
    llm=ft_llm,
    context_window=2048,  # limit the context window artifically to test refine process
)

query_engine = index.as_query_engine(service_context=ft_context)

response = query_engine.query(questions[12])

#display_response(response)

# As we can see, the fine-tuned model provides a more thorough response! This lines up with the increased faithfullness score from ragas, since the answer is more representative of the retrieved context.

# ## Conclusion
# 
# So, in conclusion, finetuning with only ~61 questions actually helped improve our eval scores!
# 
# **answer_relevancy: 0.9725 -> 0.9607**
# 
# The answer relevancy dips slightly but it's very small.
# 
# **faithfulness: 0.7325 -> 0.7917**
# 
# The faithfulness appears to have been improved! This mains the anwers given better fuffil the original question that was asked.
