

import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from dotenv import load_dotenv
load_dotenv()

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI
from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext
from read import get_filelisform

pdf_files1 = get_filelisform('./data-zh','.pdf')[:1]
pdf_files = get_filelisform('./data-zh','.txt')[:1]
print(pdf_files)

# Load all wiki documents
city_docs = []
for file in pdf_files:
    docs = SimpleDirectoryReader(
        input_files=[file]
    ).load_data()
    title = file.split(':')[0]
    docs[0].doc_id = title
    city_docs.extend(docs)


api_base1=os.environ['openai_api_base1']
api_key1=os.environ['openai_api_key1']

# LLM (gpt-3.5-turbo)
system_prompt="Always respond in Chinese"
chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo",api_base=api_base1,api_key=api_key1,system_prompt=system_prompt)
service_context = ServiceContext.from_defaults(llm=chatgpt, chunk_size=1024)

from llama_index.prompts.base import PromptTemplate


summary_query =  (
    "always respond in Chinese"
    "Describe what the provided text is about. "
    "Also describe some of the questions that this text can answer. "
)
# default mode of building the index
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True,
)
doc_summary_index = DocumentSummaryIndex.from_documents(
    city_docs,
    summary_query=summary_query,
    service_context=service_context,
    response_synthesizer=response_synthesizer,
    show_progress=True,
)

doc_summary_index.get_document_summary(pdf_files[0])

persist_dir="index-zh-4"
doc_summary_index.storage_context.persist(persist_dir)



# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
doc_summary_index = load_index_from_storage(storage_context)



query_engine = doc_summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)

response = query_engine.query("波士顿是美国哪个州的首府和最大城市？")

print(response)



