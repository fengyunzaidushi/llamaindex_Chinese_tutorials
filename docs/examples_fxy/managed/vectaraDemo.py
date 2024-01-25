#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/managed/vectaraDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Vectara Managed Index

# Vectara is the first example of a "Managed" Index, a new type of index in Llama-index which is managed via an API.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index import SimpleDirectoryReader
from llama_index.indices import VectaraIndex

# ### Loading documents
# Load the documents stored in the `Uber 10q` using the SimpleDirectoryReader

documents = SimpleDirectoryReader(os.path.abspath("../data/10q/")).load_data()
print(f"documents loaded into {len(documents)} document objects")
print(f"Document ID of first doc is {documents[0].doc_id}")

# ### Add the content of the documents into a pre-created Vectara corpus
# Here we assume an empty corpus is created and the details are available as environment variables:
# * VECTARA_CORPUS_ID
# * VECTARA_CUSTOMER_ID
# * VECTARA_API_KEY

index = VectaraIndex.from_documents(documents)

# ### Query the Vectara Index
# We can now ask questions using the VectaraIndex retriever.

query = "Is Uber still losing money or have they achieved profitability?"

# First we use the retriever to list the returned documents:

query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.retrieve(query)
texts = [t.node.text for t in response]
print("\n--\n".join(texts))

# with the as_query_engine(), we can ask questions and get the responses based on Vectara's full RAG pipeline:

query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query(query)
print(response)

# Note that the "response" object above includes both the summary text but also the source documents used to provide this response (citations)

# Vectara supports max-marginal-relevance natively in the backend, and this is available as a query mode. 
# Let's see an example of how to use MMR: We will run the same query "Is Uber still losing money or have they achieved profitability?" but this time we will use MMR where mmr_diversity_bias=1.0 which maximizes the focus on maximum diversity:

query_engine = index.as_query_engine(
    similarity_top_k=5,
    n_sentences_before=2,
    n_sentences_after=2,
    vectara_query_mode="mmr",
    mmr_k=50,
    mmr_diversity_bias=1.0,
)
response = query_engine.retrieve(query)

texts = [t.node.text for t in response]
print("\n--\n".join(texts))

# As you can see, the results in this case are much more diverse, and for example do not contain the same text more than once. The response is also better since the LLM had a more diverse set of facts to ground its response on:

query_engine = index.as_query_engine(
    similarity_top_k=5,
    n_sentences_before=2,
    n_sentences_after=2,
    summary_enabled=True,
    vectara_query_mode="mmr",
    mmr_k=50,
    mmr_diversity_bias=1.0,
)
response = query_engine.query(query)
print(response)

# So far we've used Vectara's internal summarization capability, which is the best way for most users.
# 
# You can still use Llama-Index's standard VectorStore as_query_engine() method, in which case Vectara's summarization won't be used, and you would be using an external LLM (like OpenAI's GPT-4 or similar) and a cutom prompt from LlamaIndex to generate the summart. For this option just set summary_enabled=False

query_engine = index.as_query_engine(
    similarity_top_k=5,
    summary_enabled=False,
    vectara_query_mode="mmr",
    mmr_k=50,
    mmr_diversity_bias=0.5,
)
response = query_engine.query(query)
print(response)

