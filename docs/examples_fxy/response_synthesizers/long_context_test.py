#!/usr/bin/env python
# coding: utf-8

# # Stress-Testing Long Context LLMs with a Recall Task
# 
# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/agent/openai_retrieval_benchmark.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 

# 
# Similarly, we analyze the "needle in a haystack" recall capabilities of long-context LLms. We show an incremental extension by 1) adding Claude, and 2) testing recall where context **exceeds** context window, triggering response synthesis strategies.
# 
# We use a fixed document - the 2021 Uber 10-K, which contains ~290k tokens.

import nest_asyncio

nest_asyncio.apply()

from llama_index import (
    SimpleDirectoryReader,
    Document,
    ServiceContext,
    SummaryIndex,
)
from llama_index.llms import OpenAI, Anthropic
from llama_index.evaluation import CorrectnessEvaluator

# ## Setup Data / Indexes
# 
# We load the Uber 10-k

#("mkdir -p 'data/10k/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'")

## load data
uber_docs0 = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()
uber_doc = Document(text="\n\n".join([d.get_content() for d in uber_docs0]))

# We print the number of tokens below. Note that this overflows the context window of existing LLMs, requiring response synthesis strategies.

# count the number of tokens
from llama_index.utils import globals_helper

num_tokens = len(globals_helper.tokenizer(uber_doc.get_content()))
print(f"NUM TOKENS: {num_tokens}")

# ## Try Out Different Experiments

# ### Define Context String
# 
# Here we insert a single sentence of context that we're going to "hide" within the overall document at different positions.

context_str = "Jerry's favorite snack is Hot Cheetos."
query_str = "What is Jerry's favorite snack?"

def augment_doc(doc_str, context, position):
    """Augment doc with additional context at a given position."""
    doc_str1 = doc_str[:position]
    doc_str2 = doc_str[position:]

    return f"{doc_str1}...\n\n{context}\n\n...{doc_str2}"

test_str = augment_doc(
    uber_doc.get_content(), context_str, int(0.5 * len(uber_doc.get_content()))
)

# ### Define Experiment Loop
# 
# The experiment loop is the following:
# 1. Go through the set of positions (indicated by a percentile relative to the length of the doc)
# 2. For each position, inject the context string at that position.
# 3. Load the entire doc into our `SummaryIndex`, get the corresponding query engine.
# 4. When a question is asked, we trigger response synthesis over the entire document (create-and-refine, or tree summarize).
# 5. Compare predicted response against expected response with our `CorrectnessEvaluator`

async def run_experiments(
    doc, position_percentiles, context_str, query, llm, response_mode="compact"
):
    service_context = ServiceContext.from_defaults(llm=llm)

    eval_service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-4-1106-preview")
    )
    correctness_evaluator = CorrectnessEvaluator(
        service_context=eval_service_context
    )
    eval_scores = {}
    for idx, position_percentile in enumerate(position_percentiles):
        print(f"Position percentile: {position_percentile}")
        position_idx = int(position_percentile * len(uber_doc.get_content()))
        new_doc_str = augment_doc(
            uber_doc.get_content(), context_str, position_idx
        )
        new_doc = Document(text=new_doc_str)
        index = SummaryIndex.from_documents(
            [new_doc], service_context=service_context
        )
        query_engine = index.as_query_engine(response_mode=response_mode)
        print(f"Query: {query}")

        # uncomment for async
        # response = await query_engine.aquery(query)
        response = query_engine.query(query)
        print(f"Response: {str(response)}")
        eval_result = correctness_evaluator.evaluate(
            query=query, response=str(response), reference=context_str
        )
        eval_score = eval_result.score
        print(f"Eval score: {eval_score}")
        eval_scores[position_percentile] = eval_score
    return eval_scores

position_percentiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

llm = OpenAI(model="gpt-4-1106-preview")

eval_scores_gpt4 = await run_experiments(
    [uber_doc],
    position_percentiles,
    context_str,
    query_str,
    llm,
    response_mode="compact",
)

llm = OpenAI(model="gpt-4-1106-preview")
eval_scores_gpt4_ts = await run_experiments(
    [uber_doc],
    position_percentiles,
    context_str,
    query_str,
    llm,
    response_mode="tree_summarize",
)

llm = Anthropic(model="claude-2")

eval_scores_anthropic = await run_experiments(
    [uber_doc], position_percentiles, context_str, query_str, llm
)

# NOTE: incomplete, running into timeout errors
llm = Anthropic(model="claude-2")
eval_scores_anthropic = await run_experiments(
    [uber_doc],
    position_percentiles,
    context_str,
    query_str,
    llm,
    response_mode="tree_summarize",
)

