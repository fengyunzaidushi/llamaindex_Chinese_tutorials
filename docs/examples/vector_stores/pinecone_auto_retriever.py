#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/pinecone_auto_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Auto Retriever (with Pinecone + Arize Phoenix)
# 

# 
# The steps are the following:
# 1. We'll do some setup, load data, build a Pinecone vector index.
# 2. We'll define our autoretriever and run some sample queries.
# 3. We'll use Phoenix to observe each trace and visualize the prompt inputs/outputs.
# 4. We'll show you how to customize the auto-retrieval prompt.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index scikit-learn arize-phoenix')

# ## 1. Setup Pinecone/Phoenix, Load Data, and Build Vector Index
# 

# 
# We also setup Phoenix so that it captures downstream traces.

# setup Phoenix
import phoenix as px
import llama_index

px.launch_app()
llama_index.set_global_handler("arize_phoenix")

import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import os

os.environ["PINECONE_API_KEY"] = ""

import pinecone

api_key = os.environ["PINECONE_API_KEY"]
pinecone.init(api_key=api_key, environment="gcp-starter")

# dimensions are for text-embedding-ada-002
try:
    pinecone.create_index(
        "quickstart-index", dimension=1536, metric="euclidean", pod_type="p1"
    )
except Exception as e:
    # most likely index already exists
    print(e)
    pass

pinecone_index = pinecone.Index("quickstart-index")

# Optional: delete data in your pinecone index
pinecone_index.delete(delete_all=True, namespace="test")

# #### Load documents, build the PineconeVectorStore and VectorStoreIndex

from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores import PineconeVectorStore

from llama_index.schema import TextNode

nodes = [
    TextNode(
        text="The Shawshank Redemption",
        metadata={
            "author": "Stephen King",
            "theme": "Friendship",
            "year": 1994,
        },
    ),
    TextNode(
        text="The Godfather",
        metadata={
            "director": "Francis Ford Coppola",
            "theme": "Mafia",
            "year": 1972,
        },
    ),
    TextNode(
        text="Inception",
        metadata={
            "director": "Christopher Nolan",
            "theme": "Fiction",
            "year": 2010,
        },
    ),
    TextNode(
        text="To Kill a Mockingbird",
        metadata={
            "author": "Harper Lee",
            "theme": "Mafia",
            "year": 1960,
        },
    ),
    TextNode(
        text="1984",
        metadata={
            "author": "George Orwell",
            "theme": "Totalitarianism",
            "year": 1949,
        },
    ),
    TextNode(
        text="The Great Gatsby",
        metadata={
            "author": "F. Scott Fitzgerald",
            "theme": "The American Dream",
            "year": 1925,
        },
    ),
    TextNode(
        text="Harry Potter and the Sorcerer's Stone",
        metadata={
            "author": "J.K. Rowling",
            "theme": "Fiction",
            "year": 1997,
        },
    ),
]

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    namespace="test",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(nodes, storage_context=storage_context)

# ## 2. Define Autoretriever, Run Some Sample Queries

# ### Setup the `VectorIndexAutoRetriever`
# 
# One of the inputs is a `schema` describing what content the vector store collection contains. This is similar to a table schema describing a table in the SQL database. This schema information is then injected into the prompt, which is passed to the LLM to infer what the full query should be (including metadata filters).

from llama_index.indices.vector_store.retrievers import (
    VectorIndexAutoRetriever,
)
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo

vector_store_info = VectorStoreInfo(
    content_info="famous books and movies",
    metadata_info=[
        MetadataInfo(
            name="director",
            type="str",
            description=("Name of the director"),
        ),
        MetadataInfo(
            name="theme",
            type="str",
            description=("Theme of the book/movie"),
        ),
        MetadataInfo(
            name="year",
            type="int",
            description=("Year of the book/movie"),
        ),
    ],
)
retriever = VectorIndexAutoRetriever(
    index,
    vector_store_info=vector_store_info,
    empty_query_top_k=10,
    # this is a hack to allow for blank queries in pinecone
    default_empty_query_vector=[0] * 1536,
)

# ### Let's run some queries
# 
# Let's run some sample queries that make use of the structured information.

nodes = retriever.retrieve(
    "Tell me about some books/movies after the year 2000"
)

for node in nodes:
    print(node.get_content(metadata_mode="all"))

nodes = retriever.retrieve("Tell me about some books that are Fiction")

for node in nodes:
    print(node.id_)
    print(node.get_content(metadata_mode="all"))

# #### Pass in Additional Metadata Filters
# 
# If you have additional metadata filters you want to pass in that aren't autoinferred, do the following.

from llama_index.vector_stores import MetadataFilters

filter_dicts = [{"key": "year", "operator": "==", "value": 1997}]
filters = MetadataFilters.from_dicts(filter_dicts)
retriever2 = VectorIndexAutoRetriever(
    index,
    vector_store_info=vector_store_info,
    empty_query_top_k=10,
    # this is a hack to allow for blank queries in pinecone
    default_empty_query_vector=[0] * 1536,
    extra_filters=filters,
)

nodes = retriever2.retrieve("Tell me about some books that are Fiction")
for node in nodes:
    print(node.id_)
    print(node.get_content(metadata_mode="all"))

# #### Example of a failing Query
# 
# Note that no results are retrieved! We'll fix this later on.

nodes = retriever.retrieve("Tell me about some books that are mafia-themed")

for node in nodes:
    print(node.id_)
    print(node.get_content(metadata_mode="all"))

# ### Visualize Traces
# 
# Let's open up Phoenix to take a look at the traces! 
# 
# <img src="https://drive.google.com/uc?export=view&id=1PCEwIdv7GcInk3i6ebd2WWjTp9ducG5F"/>
# 
# Let's take a look at the auto-retrieval prompt. We see that the auto-retrieval prompt makes use of two few-shot examples.

# ## Improve the Auto-retrieval Prompt
# 
# Our auto-retrieval prompt works, but it can be improved in various ways. Some examples include the fact that it includes 2 hardcoded few-shot examples (how can you include your own?), and also the fact that the auto-retrieval doesn't "always" infer the right metadata filters.
# 
# For instance, all the `theme` fields are capitalized. How do we tell the LLM that, so it doesn't erroneously infer a "theme" that's in lower-case? 
# 
# Let's take a stab at modifying the prompt! 

from llama_index.prompts import #display_prompt_dict, PromptTemplate

prompts_dict = retriever.get_prompts()

#display_prompt_dict(prompts_dict)

# look at required template variables.
prompts_dict["prompt"].template_vars

# ### Customize the Prompt
# 
# Let's customize the prompt a little bit. We do the following:
# - Take out the first few-shot example to save tokens
# - Add a message to always capitalize a letter if inferring "theme".
# 
# Note that the prompt template expects `schema_str`, `info_str`, and `query_str` to be defined.

# write prompt template, and modify it.

prompt_tmpl_str = """\
Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

{schema_str}

The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters take into account the descriptions of attributes.
Make sure that filters are only used as needed. If there are no filters that should be applied return [] for the filter value.
If the user's query explicitly mentions number of documents to retrieve, set top_k to that number, otherwise do not set top_k.

<< Example 1. >>
Data Source:
```json
{{
    "metadata_info": [
        {{
            "name": "author",
            "type": "str",
            "description": "Author name"
        }},
        {{
            "name": "book_title",
            "type": "str",
            "description": "Book title"
        }},
        {{
            "name": "year",
            "type": "int",
            "description": "Year Published"
        }},
        {{
            "name": "pages",
            "type": "int",
            "description": "Number of pages"
        }},
        {{
            "name": "summary",
            "type": "str",
            "description": "A short summary of the book"
        }}
    ],
    "content_info": "Classic literature"
}}
```

User Query:
What are some books by Jane Austen published after 1813 that explore the theme of marriage for social standing?

Additional Instructions:
None

Structured Request:
```json
{{"query": "Books related to theme of marriage for social standing", "filters": [{{"key": "year", "value": "1813", "operator": ">"}}, {{"key": "author", "value": "Jane Austen", "operator": "=="}}], "top_k": null}}

```

<< Example 2. >>
Data Source:
```json
{info_str}
```

User Query:
{query_str}

Additional Instructions:
{additional_instructions}

Structured Request:
"""

prompt_tmpl = PromptTemplate(prompt_tmpl_str)

# You'll notice we added an `additional_instructions` template variable. This allows us to insert vector collection-specific instructions. 
# 
# We'll use `partial_format` to add the instruction.

add_instrs = """\
If one of the filters is 'theme', please make sure that the first letter of the inferred value is capitalized. Only words that are capitalized are valid values for "theme". \
"""
prompt_tmpl = prompt_tmpl.partial_format(additional_instructions=add_instrs)

retriever.update_prompts({"prompt": prompt_tmpl})

# ### Re-run some queries
# 
# Now let's try rerunning some queries, and we'll see that the value is auto-inferred.

nodes = retriever.retrieve("Tell me about some books that are mafia-themed")

for node in nodes:
    print(node.id_)
    print(node.get_content(metadata_mode="all"))

