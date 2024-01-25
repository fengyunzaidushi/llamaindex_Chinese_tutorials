#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/managed/manage_retrieval_benchmark.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Semantic Retriever Benchmark
# 

# * Google Semantic Retrieval
# * LlamaIndex Retrieval
# * Vectara Managed Retrieval
# * ColBERT-V2 end-to-end Retrieval

# #

get_ipython().run_line_magic('pip', 'install llama-index')
get_ipython().run_line_magic('pip', 'install "google-ai-generativelanguage>=0.4,<=1.0"')
get_ipython().run_line_magic('pip', 'install torch sentence-transformers')

# ### Google Authentication Overview
# 
# The Google Semantic Retriever API lets you perform semantic search on your own data. Since it's **your data**, this needs stricter access controls than API Keys. Authenticate with OAuth through service accounts or through your user credentials. This quickstart uses a simplified authentication approach for a testing environment, and service account setup are typically easier to start. For a production environment, learn about [authentication and authorization](https://developers.google.com/workspace/guides/auth-overview) before choosing the [access credentials](https://developers.google.com/workspace/guides/create-credentials#choose_the_access_credential_that_is_right_for_you) that are appropriate for your app.
# 
# Demo recording for authenticating using service accounts: [Demo](https://drive.google.com/file/d/199LzrdhuuiordS15MJAxVrPKAwEJGPOh/view?usp=sharing)
# 
# **Note**: At this time, the Google Generative AI Semantic Retriever API is [only available in certain regions](https://ai.google.dev/available_regions).

# #### Authentication (Option 1): OAuth using service accounts
# 
# Google Auth [service accounts](https://cloud.google.com/iam/docs/service-account-overview) let an application authenticate to make authorized Google API calls. To OAuth using service accounts, follow the steps below:
# 
# 1. Enable the `Generative Language API`: [Documentation](https://developers.generativeai.google/tutorials/oauth_quickstart#1_enable_the_api)
# 
# 1. Create the Service Account by following the [documentation](https://developers.google.com/identity/protocols/oauth2/service-account#creatinganaccount).
# 
#  * After creating the service account, generate a service account key.
# 
# 1. Upload your service account file by using the file icon on the left sidebar, then the upload icon, as shown in the screenshot below.
# 
# <img width=400 src="https://developers.generativeai.google/tutorials/images/colab_upload.png">

get_ipython().run_line_magic('pip', 'install google-auth-oauthlib')

from google.oauth2 import service_account
from llama_index.indices.managed.google.generativeai import (
    GoogleIndex,
    set_google_config,
)

credentials = service_account.Credentials.from_service_account_file(
    "service_account_key.json",
    scopes=[
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/generative-language.retriever",
    ],
)

set_google_config(auth_credentials=credentials)

# #### Authentication (Option 2): OAuth using user credentials
# 
# Please follow [OAuth Quickstart](https://developers.generativeai.google/tutorials/oauth_quickstart) to setup OAuth using user credentials. Below are overview of steps from the documentation that are required.
# 
# 1. Enable the `Generative Language API`: [Documentation](https://developers.generativeai.google/tutorials/oauth_quickstart#1_enable_the_api)
# 
# 1. Configure the OAuth consent screen: [Documentation](https://developers.generativeai.google/tutorials/oauth_quickstart#2_configure_the_oauth_consent_screen)
# 
# 1. Authorize credentials for a desktop application: [Documentation](https://developers.generativeai.google/tutorials/oauth_quickstart#3_authorize_credentials_for_a_desktop_application)
#   * If you want to run this notebook in Colab start by uploading your
# `client_secret*.json` file using the "File > Upload" option.
# 
#  * Rename the uploaded file to `client_secret.json` or change the variable `client_file_name` in the code below.
# 
# <img width=400 src="https://developers.generativeai.google/tutorials/images/colab_upload.png">
# 
# 
# **Note**: At this time, the Google Generative AI Semantic Retriever API is [only available in certain regions](https://developers.generativeai.google/available_regions).

# Replace TODO-your-project-name with the project used in the OAuth Quickstart
project_name = "TODO-your-project-name"  #  @param {type:"string"}
# Replace TODO-your-email@gmail.com with the email added as a test user in the OAuth Quickstart
email = "ht@runllama.ai"  #  @param {type:"string"}
# Replace client_secret.json with the client_secret_* file name you uploaded.
client_file_name = "client_secret.json"

# IMPORTANT: Follow the instructions from the output - you must copy the command
# to your terminal and copy the output after authentication back here.
#('gcloud config set project $project_name')
#('gcloud config set account $email')

# NOTE: The simplified project setup in this tutorial triggers a "Google hasn't verified this app." dialog.
# This is normal, click "Advanced" -> "Go to [app name] (unsafe)"
#('gcloud auth application-default login --no-browser --client-id-file=$client_file_name --scopes="https://www.googleapis.com/auth/generative-language.retriever,https://www.googleapis.com/auth/cloud-platform"')

# This will provide you with a URL, which you should enter into your local browser.
# Follow the instruction to complete the authentication and authorization.

# ## Download Paul Graham Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Ground truth for the query `"which program did this author attend?"`
# 
# Wiki Link: https://en.wikipedia.org/wiki/Paul_Graham_(programmer)
# 
# Answer from Wiki:
# 
# ```
# Graham and his family moved to Pittsburgh, Pennsylvania in 1968, where he later attended Gateway High School. Graham gained interest in science and mathematics from his father who was a nuclear physicist.[8]
# 
# Graham received a Bachelor of Arts with a major in philosophy from Cornell University in 1986.[9][10][11] He then received a Master of Science in 1988 and a Doctor of Philosophy in 1990, both in computer science from Harvard University.[9][12]
# 
# Graham has also studied painting at the Rhode Island School of Design and at the Accademia di Belle Arti in Florence.[9][12]
# ```

# ## Google Semantic Retrieval

import os

GOOGLE_API_KEY = ""  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from llama_index import SimpleDirectoryReader
from llama_index.indices.managed.google.generativeai import GoogleIndex

# Create a Google corpus.
google_index = GoogleIndex.create_corpus(#display_name="My first corpus!")
print(f"Newly created corpus ID is {google_index.corpus_id}.")

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
google_index.insert_documents(documents)

# load Google index corpus from corpus_id
# Don't need to load it again if you have already done the ingestion step
google_index = GoogleIndex.from_corpus(corpus_id="")

# ### Google Semantic Retrieval: Using the default query engine

query_engine = google_index.as_query_engine()
response = query_engine.query("which program did this author attend?")
print(response)

# ### Show the nodes from the response

from llama_index.response.notebook_utils import #display_source_node

for r in response.source_nodes:
    #display_source_node(r, source_length=1000)

# ### Google Semantic Retrieval: Using `Verbose` Answer Style

from google.ai.generativelanguage import (
    GenerateAnswerRequest,
)

query_engine = google_index.as_query_engine(
    # Extra parameters specific to the Google query engine.
    temperature=0.3,
    answer_style=GenerateAnswerRequest.AnswerStyle.VERBOSE,
)

response = query_engine.query("Which program did this author attend?")
print(response)

from llama_index.response.notebook_utils import #display_source_node

for r in response.source_nodes:
    #display_source_node(r, source_length=1000)

# ### Google Semantic Retrieval: Using `Abstractive` Answer Style

from google.ai.generativelanguage import (
    GenerateAnswerRequest,
)

query_engine = google_index.as_query_engine(
    # Extra parameters specific to the Google query engine.
    temperature=0.3,
    answer_style=GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE,
)

response = query_engine.query("Which program did this author attend?")
print(response)

from llama_index.response.notebook_utils import #display_source_node

for r in response.source_nodes:
    #display_source_node(r, source_length=1000)

# ### Google Semantic Retrieval: Using `Extractive` Answer Style

from google.ai.generativelanguage import (
    GenerateAnswerRequest,
)

query_engine = google_index.as_query_engine(
    # Extra parameters specific to the Google query engine.
    temperature=0.3,
    answer_style=GenerateAnswerRequest.AnswerStyle.EXTRACTIVE,
)

response = query_engine.query("Which program did this author attend?")
print(response)

from llama_index.response.notebook_utils import #display_source_node

for r in response.source_nodes:
    #display_source_node(r, source_length=1000)

# ### Google Semantic Retrieval: Advanced Retrieval with LlamaIndex Reranking and Synthesizer
# * `Gemini as Reranker` LLM
# * Or using `Sentence BERT` cross encoder for Reranking
# * Adopt `Abstractive` Answer Style for Response 
# 
# For the 1st example of reranking, we tried using `Gemini` as LLM for reranking the retrieved nodes.

from llama_index.response_synthesizers.google.generativeai import (
    GoogleTextSynthesizer,
)
from llama_index.vector_stores.google.generativeai import (
    GoogleVectorStore,
    google_service_context,
)
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import Gemini
from llama_index.postprocessor import LLMRerank
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.embeddings import GeminiEmbedding

# Set up the query engine with a LLM as reranker.
response_synthesizer = GoogleTextSynthesizer.from_defaults(
    temperature=0.7, answer_style=GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE
)

embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=GOOGLE_API_KEY
)

reranker = LLMRerank(
    top_n=5,
    service_context=ServiceContext.from_defaults(
        llm=Gemini(api_key=GOOGLE_API_KEY), embed_model=embed_model
    ),
)
retriever = google_index.as_retriever(similarity_top_k=5)
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[reranker],
)

# Query for better result!
response = query_engine.query("Which program did this author attend?")

print(response.response)

# ### For the 2nd example of reranking, we use `SentenceTransformer` for cross-encoder reranking the retrieved nodes

from llama_index.postprocessor import SentenceTransformerRerank

sbert_rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=5
)

from llama_index.response_synthesizers.google.generativeai import (
    GoogleTextSynthesizer,
)
from llama_index.vector_stores.google.generativeai import (
    GoogleVectorStore,
    google_service_context,
)
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import Gemini
from llama_index.postprocessor import LLMRerank
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.embeddings import GeminiEmbedding

# Set up the query engine with a LLM as reranker.
response_synthesizer = GoogleTextSynthesizer.from_defaults(
    temperature=0.1, answer_style=GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE
)

retriever = google_index.as_retriever(similarity_top_k=5)
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[sbert_rerank],
)

# Query for better result!
response = query_engine.query("Which program did this author attend?")

print(response.response)

# ### `Observation` for `Google Semantic Retrieval`
# * `Google Semantic Retrieval` supports different `AnswerStyle`. Different style could yield different retrieval and final synthesis results. 
# * The results are mostly partly correct without reranker.
# * After applying either `Gemini as LLM` or `SBERT as cross-encoder` reranker, the results are more comprehensive and accurate.
# 
# 

# ## LlamaIndex Default Baseline with OpenAI embedding and GPT as LLM for Synthesizer 

import os

OPENAI_API_TOKEN = "sk-"
os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN

from llama_index import VectorStoreIndex, StorageContext, ServiceContext
from llama_index.vector_stores import QdrantVectorStore
from llama_index import StorageContext
import qdrant_client

# documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_retrieval_2")

vector_store = QdrantVectorStore(client=client, collection_name="collection")
qdrant_index = VectorStoreIndex.from_documents(documents)

service_context = ServiceContext.from_defaults(chunk_size=256)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

query_engine = qdrant_index.as_query_engine()
response = query_engine.query("Which program did this author attend?")
print(response)

for r in response.source_nodes:
    #display_source_node(r, source_length=1000)

# #### Rewrite the Query to include more entities related to `program`

query_engine = qdrant_index.as_query_engine()
response = query_engine.query(
    "Which universities or schools or programs did this author attend?"
)
print(response)

# ## LlamaIndex Default Configuration with LLM Reranker and Tree Summarize for Response

from llama_index import get_response_synthesizer

reranker = LLMRerank(top_n=3, service_context=service_context)
retriever = qdrant_index.as_retriever(similarity_top_k=3)
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    response_synthesizer=get_response_synthesizer(
        service_context=service_context,
        response_mode="tree_summarize",
    ),
    node_postprocessors=[reranker],
)

response = query_engine.query(
    "Which universities or schools or programs did this author attend?"
)

print(response.response)

from llama_index import get_response_synthesizer

sbert_rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=5
)
retriever = qdrant_index.as_retriever(similarity_top_k=5)
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    response_synthesizer=get_response_synthesizer(
        service_context=service_context,
        response_mode="tree_summarize",
    ),
    node_postprocessors=[sbert_rerank],
)

response = query_engine.query(
    "Which universities or schools or programs did this author attend?"
)

print(response.response)

# ### `Observation` for LlamaIndex default retrieval
# * the default query engine from LlamaIndex could only yield partly correct answer
# * With `Query Rewrite`, the results getting better.
# * With `Reranking` with top-5 retrieved results, the results get `100% accurate`.

# ## Vectara Managed Index and Retrieval

from llama_index import SimpleDirectoryReader
from llama_index.indices import VectaraIndex

vectara_customer_id = ""
vectara_corpus_id = ""
vectara_api_key = ""

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
vectara_index = VectaraIndex.from_documents(
    documents,
    vectara_customer_id=vectara_customer_id,
    vectara_corpus_id=vectara_corpus_id,
    vectara_api_key=vectara_api_key,
)

vectara_query_engine = vectara_index.as_query_engine(similarity_top_k=5)
response = vectara_query_engine.query("Which program did this author attend?")

print(response)

for r in response.source_nodes:
    #display_source_node(r, source_length=1000)

# ### `Observation` for Vectara
# * Vectara could provide somehow accurate results with citations, but it misses `Accademia di Belle Arti in Florence`.

# ## ColBERT-V2 Managed Index and Retrieval

#('git -C ColBERT/ pull || git clone https://github.com/stanford-futuredata/ColBERT.git')
import sys

sys.path.insert(0, "ColBERT/")

#('pip install faiss-cpu torch')

from llama_index import SimpleDirectoryReader, ServiceContext
from llama_index.indices import ColbertIndex
from llama_index.llms import OpenAI

import os

OPENAI_API_TOKEN = "sk-"
os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN

# ### Build ColBERT-V2 end-to-end Index

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = ColbertIndex.from_documents(
    documents=documents, service_context=service_context
)

# ### Query the ColBERT-V2 index with question

query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("Which program did this author attend?")
print(response.response)

for node in response.source_nodes:
    print(node)

response = query_engine.query(
    "Which universities or schools or programs did this author attend?"
)
print(response.response)

for node in response.source_nodes:
    print(node)

