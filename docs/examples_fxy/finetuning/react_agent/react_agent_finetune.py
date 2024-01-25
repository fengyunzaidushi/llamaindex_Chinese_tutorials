#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning a gpt-3.5 ReAct Agent on Better Chain of Thought
# 

# 
# We do this in the following steps:
# 1. Setup LlamaIndex query engine tools over Uber 10Q filings.
# 2. Use our dataset generator to generate a training/evaluation question dataset over a sample 10Q filing. Add complex variations to each question to account for multiple quarters (these complex questions help to induce chain-of-thought prompting).
# 3. Feed these questions through a GPT-4 ReAct Agent. Log inputs/outputs as a dataset to fine-tune over.
# 4. Call OpenAI fine-tuning endpoints to fine-tune gpt-3.5-turbo on this dataset.
# 5. Run qualitative evaluation: show that the fine-tuned model performs better in chain-of-thought prompting than the base model.
# 
# #### Note
# Each execution of an agent can involve multiple LLM calls through the ReAct chain-of-thought loop. The prompt inputs/output pair for each LLM call is logged as an individual datapoint in the training dataset, in the chat message format.
# 
# A big TODO here is to add more quantitative metrics for better evaluation. 

# ## Setup Data + Build Query Engine Tools
# 

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    load_index_from_storage,
)
from llama_index.llms import OpenAI

from llama_index.tools import QueryEngineTool, ToolMetadata

llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
# llm = OpenAI(temperature=0, model="gpt-4-0613")
service_context = ServiceContext.from_defaults(llm=llm)

gpt_35_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo-0613", temperature=0.3)
)
gpt4_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4-0613", temperature=0.3)
)

try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/march"
    )
    march_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/june"
    )
    june_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/sept"
    )
    sept_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False

if not index_loaded:
    # load data
    march_docs = SimpleDirectoryReader(
        input_files=["../../data/10q/uber_10q_march_2022.pdf"]
    ).load_data()
    june_docs = SimpleDirectoryReader(
        input_files=["../../data/10q/uber_10q_june_2022.pdf"]
    ).load_data()
    sept_docs = SimpleDirectoryReader(
        input_files=["../../data/10q/uber_10q_sept_2022.pdf"]
    ).load_data()

    # build index
    march_index = VectorStoreIndex.from_documents(
        march_docs, service_context=service_context
    )
    june_index = VectorStoreIndex.from_documents(
        june_docs, service_context=service_context
    )
    sept_index = VectorStoreIndex.from_documents(
        sept_docs, service_context=service_context
    )

    # persist index
    march_index.storage_context.persist(persist_dir="./storage/march")
    june_index.storage_context.persist(persist_dir="./storage/june")
    sept_index.storage_context.persist(persist_dir="./storage/sept")

march_engine = march_index.as_query_engine(
    similarity_top_k=3, service_context=service_context
)
june_engine = june_index.as_query_engine(
    similarity_top_k=3, service_context=service_context
)
sept_engine = sept_index.as_query_engine(
    similarity_top_k=3, service_context=service_context
)

from llama_index.tools.query_engine import QueryEngineTool

query_tool_sept = QueryEngineTool.from_defaults(
    query_engine=sept_engine,
    name="sept_2022",
    description=(
        f"Provides information about Uber quarterly financials ending"
        f" September 2022"
    ),
)
query_tool_june = QueryEngineTool.from_defaults(
    query_engine=june_engine,
    name="june_2022",
    description=(
        f"Provides information about Uber quarterly financials ending June"
        f" 2022"
    ),
)
query_tool_march = QueryEngineTool.from_defaults(
    query_engine=march_engine,
    name="march_2022",
    description=(
        f"Provides information about Uber quarterly financials ending March"
        f" 2022"
    ),
)

query_engine_tools = [query_tool_march, query_tool_june, query_tool_sept]

# ## Setup Base ReAct Agent (gpt-3.5-turbo)
# 
# Here we define the baseline ReAct agent over our data, on top of gpt-3.5-turbo.
# 
# We run some example queries, and show that the ReAct agent can sometimes enter the incorrect reasoning loop to answer the question.

from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-0613")
# llm = OpenAI(model="gpt-4-0613")
base_agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)

# gpt-3.5 generally gives the right response here
response = base_agent.chat(
    "Analyze Uber revenue growth over the last few quarters"
)
print(str(response))

print(str(response))

# gpt-3.5 doesn't give the right response - it doesn't first search for the quarter with the highest revenue growth
response = base_agent.chat(
    "Can you tell me about the risk factors in the quarter with the highest"
    " revenue growth?"
)
print(str(response))

# ## Generate Training/Eval Questions
# 
# Generate a synthetic dataset of questions to ask. To do this, we generate an initial set of questions over a "base" document (the March 2022 10Q), and then we use an LLM to generate variations of that question that can apply across multiple quarters. This allows us to more deeply stress-test the LLM reasoning capabilities.
# 

from llama_index.evaluation import DatasetGenerator

base_question_gen_query = (
    "You are a Teacher/ Professor. Your task is to setup a quiz/examination."
    " Using the provided context from the Uber March 10Q filing, formulate a"
    " single question that captures an important fact from the context."
    " context. Restrict the question to the context information provided."
)

dataset_generator = DatasetGenerator.from_documents(
    march_docs,
    question_gen_query=base_question_gen_query,
    service_context=gpt_35_context,
)

questions = dataset_generator.generate_questions_from_nodes(num=20)

questions

from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate

vary_question_tmpl = """\
You are a financial assistant. Given a question over a 2023 Uber 10Q filing, your goal
is to generate up to {num_vary} variations of that question that might span multiple 10Q's.

This can include compare/contrasting different 10Qs, replacing the current quarter with
another quarter, or generating questions that can only be answered over multiple quarters (be creative!)

You are given a valid set of 10Q filings. Please only generate question variations that can be
answered in that set.

For example:
Base Question: What was the free cash flow of Uber in March 2023?
Valid 10Qs: [March 2023, June 2023, September 2023]
Question Variations:
What was the free cash flow of Uber in June 2023?
Can you compare/contrast the free cash flow of Uber in June/September 2023 and offer explanations for the change?
Did the free cash flow of Uber increase of decrease in 2023?

Now let's give it a shot! 

Base Question: {base_question}
Valid 10Qs: {valid_10qs}
Question Variations:
"""

def gen_question_variations(base_questions, num_vary=3):
    """Generate question variations."""

    VALID_10Q_STR = "[March 2022, June 2022, September 2022]"

    llm = OpenAI(model="gpt-4")
    prompt_tmpl = PromptTemplate(vary_question_tmpl)

    new_questions = []
    for idx, question in enumerate(base_questions):
        new_questions.append(question)
        response = llm.complete(
            prompt_tmpl.format(
                num_vary=num_vary,
                base_question=question,
                valid_10qs=VALID_10Q_STR,
            )
        )
        # parse into newlines
        raw_lines = str(response).split("\n")
        cur_new_questions = [l for l in raw_lines if l != ""]
        print(f"[{idx}] Original Question: {question}")
        print(f"[{idx}] Generated Question Variations: {cur_new_questions}")
        new_questions.extend(cur_new_questions)

    return new_questions

def save_questions(questions, path):
    with open(path, "w") as f:
        for question in questions:
            f.write(question + "\n")

def load_questions(path):
    questions = []
    with open(path, "r") as f:
        for line in f:
            questions.append(line.strip())
    return questions

new_questions = gen_question_variations(questions)

len(new_questions)

train_questions, eval_questions = new_questions[:60], new_questions[60:]

save_questions(train_questions, "train_questions_10q.txt")
save_questions(eval_questions, "eval_questions_10q.txt")

train_questions = load_questions("train_questions_10q.txt")
eval_questions = load_questions("eval_questions_10q.txt")

# ## Use GPT-4 to Log Input/Output Pairs
# 
# We run the train questions through a GPT-4 powered ReAct agent to collect prompt outputs.
# 
# Every prompt call to the LLM is logged as an input/output pair. Since the ReAct loop can call the LLM multiple times, this means that multiple input/output pairs may be logged per user query.
# 
# Our `OpenAIFineTuningHandler` automatically collects prompt input/outputs when agent queries are run. This dataset can then be saved, in a dataset format `.jsonl` that you can directly feed to the OpenAI Finetuning endpoints.

from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.callbacks import OpenAIFineTuningHandler
from llama_index.callbacks import CallbackManager
from llama_index.agent import ReActAgent

finetuning_handler = OpenAIFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])

gpt_4_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4", temperature=0.3),
    context_window=2048,  # limit the context window artifically to test refine process
    callback_manager=callback_manager,
)

llm = OpenAI(model="gpt-4-0613")
gpt4_agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    callback_manager=callback_manager,
    verbose=True,
)

for idx, question in enumerate(train_questions):
    print(f"[{idx}] Question: {question}")
    response = gpt4_agent.query(question)
    print(f"[{idx}] Agent Response: {str(response)}")

# save events
finetuning_handler.save_finetuning_events("finetuning_events_10q.jsonl")

# ## Create `OpenAIFinetuneEngine`
# 
# We create an `OpenAIFinetuneEngine`: the finetune engine will launch a finetuning job, and returning an LLM model that you can directly plugin to the rest of LlamaIndex workflows.

from llama_index.finetuning import OpenAIFinetuneEngine

finetune_engine = OpenAIFinetuneEngine(
    "gpt-3.5-turbo",
    "finetuning_events_10q.jsonl",
    # start_job_id="<start-job-id>"  # if you have an existing job, can specify id here
)

finetune_engine.finetune()

finetune_engine.get_current_job()

ft_llm = finetune_engine.get_finetuned_model(temperature=0.3)

# ## Run Some Queries! (Compare Finetuned Agent vs. Base Agent)
# 
# We run some sample queries from the evaluation dataset over both our finetuned agent as well as the base agent.
# 
# We qualitatively look at their abilities to perform chain of thought prompting in order to arrive at the right answer.
# 
# **NOTE**: There's a big TODO to setup quantitative metrics so we can more rigorously evaluate the quality of any agent over an evaluation dataset! 

# Option 1: pass in ft_llm directly into ServiceContext
ft_context = ServiceContext.from_defaults(
    llm=ft_llm,
)

ft_agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=ft_llm,
    callback_manager=callback_manager,
    verbose=True,
)

eval_questions = []
with open("eval_questions_10q.txt", "r") as f:
    for line in f:
        eval_questions.append(line.strip())

# try a sample question
qidx = 0
print(eval_questions[qidx])

base_response = base_agent.query(eval_questions[qidx])
print(str(base_response))

ft_response = ft_agent.query(eval_questions[qidx])
print(str(ft_response))

# try the original question that failed
test_q = (
    "Can you tell me about the risk factors in the quarter with the highest"
    " revenue growth?"
)
base_response = base_agent.query(test_q)
print(str(base_response))

# NOTE: this successfully looks at each quarter for revenue growth but still falls behind GPT-4
ft_response = ft_agent.query(test_q)
print(str(ft_response))

# **Observations**: The finetuned model does much better than the base model in terms of reasoning about the current sequence of steps. It passes more detailed answers to the downstream tools and is more capable of refining its approach when initial queries don't work. This applies even if the answer isn't actually found within the context (which is a function of our automatic dataset generation capabilities). 
