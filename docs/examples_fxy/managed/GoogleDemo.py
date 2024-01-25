#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/managed/GoogleDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Google Generative Language Semantic Retriever
# 

# #

get_ipython().run_line_magic('pip', 'install llama-index')
get_ipython().run_line_magic('pip', 'install "google-ai-generativelanguage>=0.4,<=1.0"')

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
email = "TODO-your-email@gmail.com"  #  @param {type:"string"}
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

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ## Basic Usage
# 
# A `corpus` is a collection of `document`s. A `document` is a body of text that is broken into `chunk`s.

from llama_index import SimpleDirectoryReader
from llama_index.indices.managed.google.generativeai import GoogleIndex

# Create a corpus.
index = GoogleIndex.create_corpus(#display_name="My first corpus!")
print(f"Newly created corpus ID is {index.corpus_id}.")

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index.insert_documents(documents)

# Querying.
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

# Show response.
print(f"Response is {response.response}")

# Show cited passages that were used to construct the response.
for cited_text in [node.text for node in response.source_nodes]:
    print(f"Cited text: {cited_text}")

# Show answerability. 0 means not answerable from the passages.
# 1 means the model is certain the answer can be provided from the passages.
if response.metadata:
    print(
        f"Answerability: {response.metadata.get('answerable_probability', 0)}"
    )

# ## Creating a Corpus
# 
# There are various ways to create a corpus.

# The Google server will provide a corpus ID for you.
index = GoogleIndex.create_corpus(#display_name="My first corpus!")
print(index.corpus_id)

# You can also provide your own corpus ID. However, this ID needs to be globally
# unique. You will get an exception if someone else has this ID already.
index = GoogleIndex.create_corpus(
    corpus_id="my-first-corpus", #display_name="My first corpus!"
)

# If you do not provide any parameter, Google will provide ID and a default
# #display name for you.
index = GoogleIndex.create_corpus()

# ## Reusing a Corpus
# 
# Corpora you created persists on the Google servers under your account.
# You can use its ID to get a handle back.
# Then, you can query it, add more document to it, etc.

# Use a previously created corpus.
index = GoogleIndex.from_corpus(corpus_id="abc-123")

# ## Listing and Deleting Corpora
# 
# See the Python library [google-generativeai](https://github.com/google/generative-ai-python) for further documentation.

# ## Loading Documents
# 
# Many node parsers and text splitters in LlamaIndex automatically add to each node a *source_node* to associate it to a file, e.g.
# 
# ```python
#     relationships={
#         NodeRelationship.SOURCE: RelatedNodeInfo(
#             node_id="abc-123",
#             metadata={"file_name": "Title for the document"},
#         )
#     },
# ```
# 
# Both `GoogleIndex` and `GoogleVectorStore` recognize this source node,
# and will automatically create documents under your corpus on the Google servers.
# 

from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode

index = GoogleIndex.from_corpus(corpus_id="123")
index.insert_nodes(
    [
        TextNode(
            text="It was the best of times.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(
                    node_id="123",
                    metadata={"file_name": "Tale of Two Cities"},
                )
            },
        ),
        TextNode(
            text="It was the worst of times.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(
                    node_id="123",
                    metadata={"file_name": "Tale of Two Cities"},
                )
            },
        ),
        TextNode(
            text="Wassup doc",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(
                    node_id="456",
                    metadata={"file_name": "Bugs Bunny Adventure"},
                )
            },
        ),
    ]
)

# If your nodes do not have a source node, then Google server will put your nodes in a default document under your corpus.

# ## Listing and Deleting Documents
# 
# See the Python library [google-generativeai](https://github.com/google/generative-ai-python) for further documentation.

# ## Querying Corpus
# 
# Google's query engine is backed by a specially tuned LLM that grounds its response based on retrieved passages. For each response, an *answerability probability* is returned to indicate how confident the LLM was in answering the question from the retrieved passages.
# 
# Furthermore, Google's query engine supports *answering styles*, such as `ABSTRACTIVE` (succint but abstract), `EXTRACTIVE` (very brief and extractive) and `VERBOSE` (extra details).
# 
# The engine also supports *safety settings*.
# 

from google.ai.generativelanguage import (
    GenerateAnswerRequest,
    HarmCategory,
    SafetySetting,
)

index = GoogleIndex.from_corpus(corpus_id="123")
query_engine = index.as_query_engine(
    # Extra parameters specific to the Google query engine.
    temperature=0.7,
    answer_style=GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE,
    safety_setting=[
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_VIOLENCE,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
    ],
)

response = query_engine.query("What movie should I watch with my family?")

# See the Python library [google-generativeai](https://github.com/google/generative-ai-python) for further documentation.

# #

from llama_index.response.schema import Response

response = query_engine.query("What movie should I watch with my family?")
assert isinstance(response, Response)

# Show response.
print(f"Response is {response.response}")

# Show cited passages that were used to construct the response.
for cited_text in [node.text for node in response.source_nodes]:
    print(f"Cited text: {cited_text}")

# Show answerability. 0 means not answerable from the passages.
# 1 means the model is certain the answer can be provided from the passages.
print(f"Answerability: {response.metadata.get("answerable_probability", 0)}")

# ## Advanced RAG
# 
# The `GoogleIndex` is built based on `GoogleVectorStore` and `GoogleTextSynthesizer`.
# These components can be combined with other powerful constructs in LlamaIndex to produce advanced RAG applications.
# 
# Below we show a few examples.

# ### Reranker + Google Retriever
# 
# Converting content into vectors is a lossy process. LLM-based Reranking
# remediates this by reranking the retrieved content using LLM, which has higher
# fidelity because it has access to both the actual query and the passage.

from llama_index.response_synthesizers.google.generativeai import (
    GoogleTextSynthesizer,
)
from llama_index.vector_stores.google.generativeai import (
    GoogleVectorStore,
    google_service_context,
)
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import PaLM
from llama_index.postprocessor import LLMRerank
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever

# Set up the query engine with a reranker.
store = GoogleVectorStore.from_corpus(corpus_id="some-corpus-id")
index = VectorStoreIndex.from_vector_store(
    vector_store=store, service_context=google_service_context
)
response_synthesizer = GoogleTextSynthesizer.from_defaults(
    temperature=0.7, answer_style=GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE
)
reranker = LLMRerank(
    top_n=10, service_context=ServiceContext.from_defaults(llm=PaLM())
)
query_engine = RetrieverQueryEngine.from_args(
    retriever=VectorIndexRetriever(
        index=index,
        similarity_top_k=20,
    ),
    response_synthesizer=response_synthesizer,
    node_postprocessors=[reranker],
)

# Query for better result!
response = query_engine.query("What movie should I watch with my family?")

# ### Multi-Query + Google Retriever
# 
# Sometimes, a user's query can be too complex. You may get better retrieval result if you break down the original query into smaller, better focused queries.

from llama_index.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.query_engine.multistep_query_engine import (
    MultiStepQueryEngine,
)

# Set up the query engine with multi-turn query-rewriter.
store = GoogleVectorStore.from_corpus(corpus_id="some-corpus-id")
index = VectorStoreIndex.from_vector_store(
    vector_store=store, service_context=google_service_context
)
response_synthesizer = GoogleTextSynthesizer.from_defaults(
    temperature=0.7, answer_style=GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE
)
single_step_query_engine = index.as_query_engine(
    response_synthesizer=response_synthesizer
)
step_decompose_transform = StepDecomposeQueryTransform(
    llm=PaLM(), verbose=True
)
query_engine = MultiStepQueryEngine(
    query_engine=single_step_query_engine,
    query_transform=step_decompose_transform,
    response_synthesizer=response_synthesizer,
    index_summary="Ask me anything.",
    num_steps=6,
)

# Query for better result!
response = query_engine.query("What movie should I watch with my family?")

# ### HyDE + Google Retriever
# 
# When you can write prompt that would produce fake answers that share many traits
# with the real answer, you can try HyDE!

from llama_index.indices.query.query_transform import HyDEQueryTransform
from llama_index.query_engine.transform_query_engine import (
    TransformQueryEngine,
)

# Set up the query engine with multi-turn query-rewriter.
store = GoogleVectorStore.from_corpus(corpus_id="some-corpus-id")
index = VectorStoreIndex.from_vector_store(
    vector_store=store, service_context=google_service_context
)
response_synthesizer = GoogleTextSynthesizer.from_defaults(
    temperature=0.7, answer_style=GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE
)
base_query_engine = index.as_query_engine(
    response_synthesizer=response_synthesizer
)
hyde = HyDEQueryTransform(include_original=False)
hyde_query_engine = TransformQueryEngine(base_query_engine, hyde)

# Query for better result!
response = hyde_query_engine.query("What movie should I watch with my family?")

# ### Multi-Query + Reranker + HyDE + Google Retriever
# 
# Or combine them all!

# Google's retriever and AQA model setup.
store = GoogleVectorStore.from_corpus(corpus_id="some-corpus-id")
index = VectorStoreIndex.from_vector_store(
    vector_store=store, service_context=google_service_context
)
response_synthesizer = GoogleTextSynthesizer.from_defaults(
    temperature=0.7, answer_style=GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE
)

# Reranker setup.
reranker = LLMRerank(
    top_n=10, service_context=ServiceContext.from_defaults(llm=PaLM())
)
single_step_query_engine = index.as_query_engine(
    response_synthesizer=response_synthesizer, node_postprocessors=[reranker]
)

# HyDE setup.
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(single_step_query_engine, hyde)

# Multi-query setup.
step_decompose_transform = StepDecomposeQueryTransform(
    llm=PaLM(), verbose=True
)
query_engine = MultiStepQueryEngine(
    query_engine=hyde_query_engine,
    query_transform=step_decompose_transform,
    response_synthesizer=response_synthesizer,
    index_summary="Ask me anything.",
    num_steps=6,
)

# Query for better result!
response = query_engine.query("What movie should I watch with my family?")

