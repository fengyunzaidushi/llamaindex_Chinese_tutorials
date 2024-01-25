#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/metadata_extraction/MetadataExtractionSEC.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Extracting Metadata for Better Document Indexing and Understanding
# 

# 
# To combat this, we use LLMs to extract certain contextual information relevant to the document to better help the retrieval and language models disambiguate similar-looking passages.
# 
# We do this through our brand-new `Metadata Extractor` modules.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import nest_asyncio

nest_asyncio.apply()

import os
import openai

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.schema import MetadataMode

llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=512)

# We create a node parser that extracts the document title and hypothetical question embeddings relevant to the document chunk.
# 
# We also show how to instantiate the `SummaryExtractor` and `KeywordExtractor`, as well as how to create your own custom extractor 
# based on the `BaseExtractor` base class

from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    EntityExtractor,
    BaseExtractor,
)
from llama_index.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128
)

class CustomExtractor(BaseExtractor):
    def extract(self, nodes):
        metadata_list = [
            {
                "custom": (
                    node.metadata["document_title"]
                    + "\n"
                    + node.metadata["excerpt_keywords"]
                )
            }
            for node in nodes
        ]
        return metadata_list

extractors = [
    TitleExtractor(nodes=5, llm=llm),
    QuestionsAnsweredExtractor(questions=3, llm=llm),
    # EntityExtractor(prediction_threshold=0.5),
    # SummaryExtractor(summaries=["prev", "self"], llm=llm),
    # KeywordExtractor(keywords=10, llm=llm),
    # CustomExtractor()
]

transformations = [text_splitter] + extractors

from llama_index import SimpleDirectoryReader

# We first load the 10k annual SEC report for Uber and Lyft for the years 2019 and 2020 respectively.

#('mkdir -p data')
#('wget -O "data/10k-132.pdf" "https://www.dropbox.com/scl/fi/6dlqdk6e2k1mjhi8dee5j/uber.pdf?rlkey=2jyoe49bg2vwdlz30l76czq6g&dl=1"')
#('wget -O "data/10k-vFinal.pdf" "https://www.dropbox.com/scl/fi/qn7g3vrk5mqb18ko4e5in/lyft.pdf?rlkey=j6jxtjwo8zbstdo4wz3ns8zoj&dl=1"')

# Note the uninformative document file name, which may be a common scenario in a production setting
uber_docs = SimpleDirectoryReader(input_files=["data/10k-132.pdf"]).load_data()
uber_front_pages = uber_docs[0:3]
uber_content = uber_docs[63:69]
uber_docs = uber_front_pages + uber_content

from llama_index.ingestion import IngestionPipeline

pipeline = IngestionPipeline(transformations=transformations)

uber_nodes = pipeline.run(documents=uber_docs)

uber_nodes[1].metadata

# Note the uninformative document file name, which may be a common scenario in a production setting
lyft_docs = SimpleDirectoryReader(
    input_files=["data/10k-vFinal.pdf"]
).load_data()
lyft_front_pages = lyft_docs[0:3]
lyft_content = lyft_docs[68:73]
lyft_docs = lyft_front_pages + lyft_content

from llama_index.ingestion import IngestionPipeline

pipeline = IngestionPipeline(transformations=transformations)

lyft_nodes = pipeline.run(documents=lyft_docs)

lyft_nodes[2].metadata

# Since we are asking fairly sophisticated questions, we utilize a subquestion query engine for all QnA pipelines below, and prompt it to pay more attention to the relevance of the retrieved sources. 

from llama_index.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.question_gen.prompts import DEFAULT_SUB_QUESTION_PROMPT_TMPL

service_context = ServiceContext.from_defaults(
    llm=llm, text_splitter=text_splitter
)
question_gen = LLMQuestionGenerator.from_defaults(
    service_context=service_context,
    prompt_template_str="""
        Follow the example, but instead of giving a question, always prefix the question 
        with: 'By first identifying and quoting the most relevant sources, '. 
        """
    + DEFAULT_SUB_QUESTION_PROMPT_TMPL,
)

# ## Querying an Index With No Extra Metadata

from copy import deepcopy

nodes_no_metadata = deepcopy(uber_nodes) + deepcopy(lyft_nodes)
for node in nodes_no_metadata:
    node.metadata = {
        k: node.metadata[k]
        for k in node.metadata
        if k in ["page_label", "file_name"]
    }
print(
    "LLM sees:\n",
    (nodes_no_metadata)[9].get_content(metadata_mode=MetadataMode.LLM),
)

from llama_index import VectorStoreIndex
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata

index_no_metadata = VectorStoreIndex(
    nodes=nodes_no_metadata,
    service_context=ServiceContext.from_defaults(llm=OpenAI(model="gpt-4")),
)
engine_no_metadata = index_no_metadata.as_query_engine(
    similarity_top_k=10,
)

final_engine_no_metadata = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[
        QueryEngineTool(
            query_engine=engine_no_metadata,
            metadata=ToolMetadata(
                name="sec_filing_documents",
                description="financial information on companies",
            ),
        )
    ],
    question_gen=question_gen,
    use_async=True,
)

response_no_metadata = final_engine_no_metadata.query(
    """
    What was the cost due to research and development v.s. sales and marketing for uber and lyft in 2019 in millions of USD?
    Give your answer as a JSON.
    """
)
print(response_no_metadata.response)
# Correct answer:
# {"Uber": {"Research and Development": 4836, "Sales and Marketing": 4626},
#  "Lyft": {"Research and Development": 1505.6, "Sales and Marketing": 814 }}

# **RESULT**: As we can see, the QnA agent does not seem to know where to look for the right documents. As a result it gets the Lyft and Uber data completely mixed up.

# ## Querying an Index With Extracted Metadata

print(
    "LLM sees:\n",
    (uber_nodes + lyft_nodes)[9].get_content(metadata_mode=MetadataMode.LLM),
)

index = VectorStoreIndex(
    nodes=uber_nodes + lyft_nodes,
    service_context=ServiceContext.from_defaults(llm=OpenAI(model="gpt-4")),
)
engine = index.as_query_engine(
    similarity_top_k=10,
)

final_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[
        QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name="sec_filing_documents",
                description="financial information on companies.",
            ),
        )
    ],
    question_gen=question_gen,
    use_async=True,
)

response = final_engine.query(
    """
    What was the cost due to research and development v.s. sales and marketing for uber and lyft in 2019 in millions of USD?
    Give your answer as a JSON.
    """
)
print(response.response)
# Correct answer:
# {"Uber": {"Research and Development": 4836, "Sales and Marketing": 4626},
#  "Lyft": {"Research and Development": 1505.6, "Sales and Marketing": 814 }}

# **RESULT**: As we can see, the LLM answers the questions correctly.

# ### Challenges Identified in the Problem Domain
# 

# 

# 
# Other valid steps may include utilizing models that are fine-tuned on financial datasets such as Bloomberg GPT.
# 
# Finally, we can help to further enrich the metadata by providing more contextual information regarding the surrounding context that the chunk is located in.
# 
# ### Improvements to this Example
# Generally, this example can be improved further with more rigorous evaluation of both the metadata extraction accuracy, and the accuracy and recall of the QnA pipeline. Further, incorporating a larger set of documents as well as the full length documents, which may provide more confounding passages that are difficult to disambiguate, could further stresss test the system we have built and suggest further improvements. 
